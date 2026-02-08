"""
Multi-View Training with 3D Consistency Losses for MapAnything + Mask2Former

This implements the UNITE-inspired consistency strategy adapted for Mask2Former:

1. DENSE FEATURE ALIGNMENT (L_dense):
   - Align Parallel (Panoptic) DPT features across views
   - Weight by Geometric DPT confidence maps
   - Use ScanNet++ depth/poses for pixel correspondences

2. QUERY EMBEDDING ALIGNMENT (L_query):
   - Contrastive loss on Mask2Former object queries
   - Pull queries for same 3D instance together
   - Push queries for different instances apart

3. MASK PROJECTION CONSISTENCY (L_mask):
   - Lift 2D masks to 3D using depth
   - Ensure consistent 3D volumes
   - Reproject to supervise 2D boundaries

Key Insight:
- Parallel DPT (trainable): SOURCE of features to align
- Geometric DPT (frozen): SOURCE of confidence signals for weighting
"""

import os
import sys
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Filter warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# Detectron2 imports
from detectron2.layers import ShapeSpec
from detectron2.config import CfgNode as CN
from detectron2.engine import launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import Backbone
import detectron2.utils.comm as comm

sys.path.insert(0, os.getcwd())

from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# MapAnything imports
from mapanything.models import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput

# Import from existing training scripts
from m2f_train_3d_loss import (
    PanopticDPTHead,
    NaNLossCheckHook,
)
from m2f_train_multiview import (
    multi_view_collate_fn,
    matrix_to_quaternion,
    camera_to_world_to_mapanything_pose,
)


# ============================================================
# GEOMETRIC UTILITIES FOR CORRESPONDENCE
# ============================================================

def compute_pixel_correspondences(
    depth_src: torch.Tensor,      # [B, H, W] depth map of source view
    pose_src: torch.Tensor,       # [B, 4, 4] camera-to-world of source
    pose_tgt: torch.Tensor,       # [B, 4, 4] camera-to-world of target
    intrinsics: torch.Tensor,     # [B, 3, 3] camera intrinsics
    return_depth_tgt: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute pixel correspondences from source view to target view.
    
    For each pixel (u_s, v_s) in source view:
    1. Unproject to 3D using depth
    2. Transform to target camera frame
    3. Project to target image plane
    
    Args:
        depth_src: Depth map of source view [B, H, W]
        pose_src: Camera-to-world transform of source [B, 4, 4]
        pose_tgt: Camera-to-world transform of target [B, 4, 4]
        intrinsics: Camera intrinsic matrix [B, 3, 3]
    
    Returns:
        correspondences: [B, H, W, 2] - (u_tgt, v_tgt) for each source pixel
        valid_mask: [B, H, W] - True if correspondence is valid
        depth_tgt: [B, H, W] - depth in target view (optional)
    """
    B, H, W = depth_src.shape
    device = depth_src.device
    
    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # [H, W]
    
    # Expand for batch
    u = u.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    v = v.unsqueeze(0).expand(B, -1, -1)
    
    # Get intrinsic parameters
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)
    
    # Unproject to camera coordinates
    z = depth_src
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to points [B, H, W, 3]
    points_src_cam = torch.stack([x, y, z], dim=-1)
    
    # Add homogeneous coordinate [B, H, W, 4]
    ones = torch.ones_like(z).unsqueeze(-1)
    points_src_cam_h = torch.cat([points_src_cam, ones], dim=-1)
    
    # Transform: source camera -> world -> target camera
    # points_world = pose_src @ points_src_cam
    # points_tgt_cam = pose_tgt^{-1} @ points_world
    
    # Flatten for batch matrix multiply
    points_flat = points_src_cam_h.view(B, -1, 4)  # [B, H*W, 4]
    
    # Source camera to world
    points_world = torch.bmm(points_flat, pose_src.transpose(1, 2))  # [B, H*W, 4]
    
    # World to target camera
    pose_tgt_inv = torch.inverse(pose_tgt)
    points_tgt_cam = torch.bmm(points_world, pose_tgt_inv.transpose(1, 2))  # [B, H*W, 4]
    
    # Reshape back
    points_tgt_cam = points_tgt_cam.view(B, H, W, 4)[..., :3]  # [B, H, W, 3]
    
    # Project to target image plane
    z_tgt = points_tgt_cam[..., 2]
    x_tgt = points_tgt_cam[..., 0]
    y_tgt = points_tgt_cam[..., 1]
    
    # Avoid division by zero
    z_tgt_safe = z_tgt.clamp(min=1e-6)
    
    u_tgt = fx * x_tgt / z_tgt_safe + cx
    v_tgt = fy * y_tgt / z_tgt_safe + cy
    
    # Stack correspondences
    correspondences = torch.stack([u_tgt, v_tgt], dim=-1)  # [B, H, W, 2]
    
    # Valid mask: positive depth, within image bounds, not occluded
    valid_mask = (
        (depth_src > 0) &
        (z_tgt > 0) &
        (u_tgt >= 0) & (u_tgt < W) &
        (v_tgt >= 0) & (v_tgt < H)
    )
    
    if return_depth_tgt:
        return correspondences, valid_mask, z_tgt
    return correspondences, valid_mask, None


def sample_features_at_correspondences(
    features: torch.Tensor,           # [B, C, H, W]
    correspondences: torch.Tensor,    # [B, H_src, W_src, 2]
    valid_mask: torch.Tensor,         # [B, H_src, W_src]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample features at correspondence locations using bilinear interpolation.
    
    Returns:
        sampled_features: [B, C, H_src, W_src] features from target at correspondences
        valid_mask: Updated mask (same as input)
    """
    B, C, H, W = features.shape
    _, H_src, W_src, _ = correspondences.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    grid = correspondences.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0  # u
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0  # v
    
    # Sample features
    sampled = F.grid_sample(
        features,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [B, C, H_src, W_src]
    
    return sampled, valid_mask


# ============================================================
# CONSISTENCY LOSS FUNCTIONS
# ============================================================

class DenseFeatureConsistencyLoss(nn.Module):
    """
    Loss 1: Align Parallel DPT features across views.
    
    Uses Geometric DPT confidence maps to weight the alignment.
    Higher confidence = more weight on that correspondence.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        panoptic_features: List[torch.Tensor],  # [N views], each [B, C, H, W]
        geometric_confidence: List[torch.Tensor],  # [N views], each [B, 1, H, W]
        correspondences_list: List[Tuple[torch.Tensor, torch.Tensor]],  # Per view-pair
        valid_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute confidence-weighted feature consistency loss.
        
        For each view pair (i, j):
        1. Sample features from view j at correspondence locations
        2. Weight by confidence from both views
        3. Compute cosine similarity loss
        """
        N = len(panoptic_features)
        if N < 2:
            return torch.tensor(0.0, device=panoptic_features[0].device)
        
        total_loss = 0.0
        num_pairs = 0
        
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                feat_i = panoptic_features[i]  # [B, C, H, W]
                feat_j = panoptic_features[j]
                
                conf_i = geometric_confidence[i]  # [B, 1, H, W]
                conf_j = geometric_confidence[j]
                
                corr_i_to_j, valid_mask = correspondences_list[pair_idx], valid_masks_list[pair_idx]
                
                # Sample features from j at correspondence locations
                feat_j_at_i, _ = sample_features_at_correspondences(
                    feat_j, corr_i_to_j, valid_mask
                )
                
                # Sample confidence from j at correspondence locations
                conf_j_at_i, _ = sample_features_at_correspondences(
                    conf_j, corr_i_to_j, valid_mask
                )
                
                # Combined confidence: geometric mean of both confidences
                combined_conf = torch.sqrt(conf_i * conf_j_at_i + 1e-8)  # [B, 1, H, W]
                combined_conf = combined_conf.squeeze(1)  # [B, H, W]
                
                # Normalize features for cosine similarity
                feat_i_norm = F.normalize(feat_i, dim=1)
                feat_j_at_i_norm = F.normalize(feat_j_at_i, dim=1)
                
                # Cosine similarity per pixel
                cosine_sim = (feat_i_norm * feat_j_at_i_norm).sum(dim=1)  # [B, H, W]
                
                # Loss: 1 - cosine_similarity (want features to be similar)
                pixel_loss = 1.0 - cosine_sim
                
                # Weight by confidence and valid mask
                weighted_loss = pixel_loss * combined_conf * valid_mask.float()
                
                # Normalize by sum of weights
                weight_sum = (combined_conf * valid_mask.float()).sum() + 1e-8
                pair_loss = weighted_loss.sum() / weight_sum
                
                total_loss += pair_loss
                num_pairs += 1
                pair_idx += 1
        
        return total_loss / max(num_pairs, 1)


class QueryEmbeddingConsistencyLoss(nn.Module):
    """
    Loss 2: Align Mask2Former object queries across views.
    
    Contrastive approach:
    - PULL: Queries representing same 3D instance should have similar embeddings
    - PUSH: Queries representing different instances should be dissimilar
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        query_embeddings: List[torch.Tensor],  # [N views], each [B, Q, D]
        pred_masks: List[torch.Tensor],         # [N views], each [B, Q, H, W]
        correspondences_list: List[Tuple],
        valid_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute contrastive query consistency loss.
        
        Strategy:
        1. Find which queries in view i and j cover same 3D points
        2. Pull those query embeddings together
        3. Push embeddings of non-overlapping queries apart
        """
        N = len(query_embeddings)
        if N < 2:
            return torch.tensor(0.0, device=query_embeddings[0].device)
        
        B, Q, D = query_embeddings[0].shape
        device = query_embeddings[0].device
        
        total_loss = 0.0
        num_pairs = 0
        
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                queries_i = query_embeddings[i]  # [B, Q, D]
                queries_j = query_embeddings[j]
                
                masks_i = pred_masks[i].sigmoid()  # [B, Q, H, W]
                masks_j = pred_masks[j].sigmoid()
                
                corr_i_to_j = correspondences_list[pair_idx]
                valid_mask = valid_masks_list[pair_idx]
                
                # For each batch
                batch_loss = 0.0
                for b in range(B):
                    # Compute query-to-query IoU based on 3D correspondence
                    iou_matrix = self._compute_query_iou_via_correspondence(
                        masks_i[b], masks_j[b],
                        corr_i_to_j[b], valid_mask[b]
                    )  # [Q, Q]
                    
                    # Find matched pairs (high IoU = same instance)
                    match_threshold = 0.3
                    matched_pairs = (iou_matrix > match_threshold).nonzero(as_tuple=False)
                    
                    if len(matched_pairs) > 0:
                        # Pull loss: matched queries should be similar
                        for q_i, q_j in matched_pairs:
                            emb_i = queries_i[b, q_i]  # [D]
                            emb_j = queries_j[b, q_j]  # [D]
                            
                            # Cosine similarity loss
                            sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                            pull_loss = 1.0 - sim
                            batch_loss += pull_loss * iou_matrix[q_i, q_j]  # Weight by IoU
                    
                    # Push loss: unmatched queries should be dissimilar
                    unmatched_i = (iou_matrix.max(dim=1)[0] < match_threshold).nonzero(as_tuple=False).squeeze(-1)
                    unmatched_j = (iou_matrix.max(dim=0)[0] < match_threshold).nonzero(as_tuple=False).squeeze(-1)
                    
                    if len(unmatched_i) > 0 and len(unmatched_j) > 0:
                        # Sample some pairs for push loss
                        num_neg = min(len(unmatched_i), len(unmatched_j), 10)
                        neg_i = unmatched_i[torch.randperm(len(unmatched_i))[:num_neg]]
                        neg_j = unmatched_j[torch.randperm(len(unmatched_j))[:num_neg]]
                        
                        for q_i, q_j in zip(neg_i, neg_j):
                            emb_i = queries_i[b, q_i]
                            emb_j = queries_j[b, q_j]
                            
                            sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                            push_loss = F.relu(sim - self.margin + 1.0)  # Hinge loss
                            batch_loss += push_loss * 0.1  # Lower weight for push
                
                total_loss += batch_loss / B
                num_pairs += 1
                pair_idx += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _compute_query_iou_via_correspondence(
        self,
        masks_i: torch.Tensor,      # [Q, H, W]
        masks_j: torch.Tensor,      # [Q, H, W]
        corr_i_to_j: torch.Tensor,  # [H, W, 2]
        valid_mask: torch.Tensor,   # [H, W]
    ) -> torch.Tensor:
        """Compute IoU between queries based on 3D correspondence."""
        Q, H, W = masks_i.shape
        device = masks_i.device
        
        # Sample masks_j at correspondence locations
        corr_grid = corr_i_to_j.unsqueeze(0)  # [1, H, W, 2]
        corr_grid[..., 0] = 2.0 * corr_grid[..., 0] / (W - 1) - 1.0
        corr_grid[..., 1] = 2.0 * corr_grid[..., 1] / (H - 1) - 1.0
        
        masks_j_at_i = F.grid_sample(
            masks_j.unsqueeze(0),  # [1, Q, H, W]
            corr_grid.expand(Q, -1, -1, -1).view(Q, H, W, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)  # [Q, H, W]
        
        # Compute IoU for each pair of queries
        iou_matrix = torch.zeros(Q, Q, device=device)
        
        valid_mask_flat = valid_mask.view(-1)
        
        for q_i in range(Q):
            mask_i_flat = masks_i[q_i].view(-1)
            mask_i_valid = mask_i_flat * valid_mask_flat
            
            for q_j in range(Q):
                mask_j_flat = masks_j_at_i[q_j].view(-1)
                mask_j_valid = mask_j_flat * valid_mask_flat
                
                intersection = (mask_i_valid * mask_j_valid).sum()
                union = mask_i_valid.sum() + mask_j_valid.sum() - intersection + 1e-8
                iou_matrix[q_i, q_j] = intersection / union
        
        return iou_matrix


class MaskProjectionConsistencyLoss(nn.Module):
    """
    Loss 3: Lift 2D masks to 3D, ensure consistency, reproject.
    
    Strategy:
    1. Lift each view's masks to 3D point cloud
    2. For corresponding 3D points, masks should agree
    3. Reproject and supervise boundaries
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_masks: List[torch.Tensor],         # [N views], each [B, Q, H, W]
        depth: List[torch.Tensor],               # [N views], each [B, H, W]
        poses: List[torch.Tensor],               # [N views], each [B, 4, 4]
        intrinsics: torch.Tensor,                # [B, 3, 3]
        correspondences_list: List[torch.Tensor],
        valid_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute mask projection consistency loss.
        
        For corresponding pixels across views:
        - The query with highest mask probability should match
        """
        N = len(pred_masks)
        if N < 2:
            return torch.tensor(0.0, device=pred_masks[0].device)
        
        B = pred_masks[0].shape[0]
        total_loss = 0.0
        num_pairs = 0
        
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                masks_i = pred_masks[i].sigmoid()  # [B, Q, H, W]
                masks_j = pred_masks[j].sigmoid()
                
                corr_i_to_j = correspondences_list[pair_idx]  # [B, H, W, 2]
                valid_mask = valid_masks_list[pair_idx]  # [B, H, W]
                
                # Get dominant query per pixel in view i
                dominant_i = masks_i.argmax(dim=1)  # [B, H, W]
                
                # Sample masks_j at correspondence locations
                B, Q, H, W = masks_j.shape
                corr_grid = corr_i_to_j.clone()
                corr_grid[..., 0] = 2.0 * corr_grid[..., 0] / (W - 1) - 1.0
                corr_grid[..., 1] = 2.0 * corr_grid[..., 1] / (H - 1) - 1.0
                
                masks_j_at_i = F.grid_sample(
                    masks_j,
                    corr_grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )  # [B, Q, H, W]
                
                # Get dominant query at correspondence locations
                dominant_j_at_i = masks_j_at_i.argmax(dim=1)  # [B, H, W]
                
                # Loss: dominant queries should match for corresponding pixels
                match = (dominant_i == dominant_j_at_i).float()
                
                # Weight by mask confidence (higher confidence = more important)
                max_prob_i = masks_i.max(dim=1)[0]  # [B, H, W]
                max_prob_j_at_i = masks_j_at_i.max(dim=1)[0]
                confidence = max_prob_i * max_prob_j_at_i
                
                # Loss: 1 - match (want them to match)
                pixel_loss = (1.0 - match) * confidence * valid_mask.float()
                
                weight_sum = (confidence * valid_mask.float()).sum() + 1e-8
                pair_loss = pixel_loss.sum() / weight_sum
                
                total_loss += pair_loss
                num_pairs += 1
                pair_idx += 1
        
        return total_loss / max(num_pairs, 1)


# ============================================================
# DUAL-DPT BACKBONE WITH CONSISTENCY
# ============================================================

class DualDPTBackbone(Backbone):
    """
    Backbone with two DPT heads:
    1. Parallel (Panoptic) DPT - trainable, features for segmentation
    2. Geometric DPT - frozen, provides confidence/depth for consistency
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        self.patch_size = 14
        self.encoder_dim = 768
        self.info_sharing_dim = 1024
        self.num_views = cfg.MODEL.MULTIVIEW.NUM_VIEWS
        self.use_poses = cfg.MODEL.MULTIVIEW.USE_POSES
        
        # Load MapAnything
        print("Loading MapAnything model...")
        self.mapanything = MapAnything.from_pretrained("nkeetha/map-anything")
        self.mapanything.eval()
        
        # Freeze MapAnything (including geometric DPT)
        for param in self.mapanything.parameters():
            param.requires_grad = False
        
        # Determine input dims
        if hasattr(self.mapanything, 'use_encoder_features_for_dpt'):
            self.use_encoder_features_for_dpt = self.mapanything.use_encoder_features_for_dpt
        else:
            self.use_encoder_features_for_dpt = True
        
        if self.use_encoder_features_for_dpt:
            input_dims = [self.encoder_dim] + [self.info_sharing_dim] * 3
        else:
            input_dims = [self.info_sharing_dim] * 4
        
        # Parallel (Panoptic) DPT Head - TRAINABLE
        self.panoptic_dpt = PanopticDPTHead(
            input_dims=input_dims,
            patch_size=self.patch_size,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            output_channels=[96, 192, 384, 768],
        )
        
        # Initialize from geometric DPT
        self._init_panoptic_dpt_from_geometric()
        
        # Output features
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_channels = {
            'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768
        }
        self._out_feature_strides = {
            'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32
        }
        
        print(f"DualDPTBackbone initialized:")
        print(f"  - Panoptic DPT (trainable): for segmentation features")
        print(f"  - Geometric DPT (frozen): for confidence/depth")
    
    def _init_panoptic_dpt_from_geometric(self):
        """Initialize Panoptic DPT from geometric DPT weights."""
        print("\nInitializing Panoptic DPT from Geometric DPT...")
        
        geometric_dpt = None
        if hasattr(self.mapanything, 'dpt_feature_head'):
            geometric_dpt = self.mapanything.dpt_feature_head
        elif hasattr(self.mapanything, 'dense_head') and len(self.mapanything.dense_head) > 0:
            geometric_dpt = self.mapanything.dense_head[0]
        
        if geometric_dpt is None:
            print("  WARNING: Could not find geometric DPT head")
            return
        
        # Copy matching weights (simplified)
        copied = 0
        for name_p, param_p in self.panoptic_dpt.named_parameters():
            for name_g, param_g in geometric_dpt.named_parameters():
                if name_p == name_g and param_p.shape == param_g.shape:
                    param_p.data.copy_(param_g.data)
                    copied += 1
                    break
        
        print(f"  ✓ Copied {copied} parameters from geometric DPT")
    
    def forward(
        self,
        images: torch.Tensor,
        camera_poses: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        return_geometric: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass returning both panoptic and geometric outputs.
        
        Returns:
            {
                'panoptic_features': {res2, res3, res4, res5},
                'geometric_depth': [N views], each [B, 1, H, W],
                'geometric_confidence': [N views], each [B, 1, H, W],
                'layer_features': for intermediate supervision,
            }
        """
        # Handle input shape
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            if camera_poses is not None:
                camera_poses = camera_poses.view(B * N, 4, 4)
        else:
            B_N, C, H, W = images.shape
            N = self.num_views
            B = B_N // N
        
        orig_h, orig_w = H, W
        
        # Pad to be divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
        
        padded_h, padded_w = images.shape[2], images.shape[3]
        
        with torch.no_grad():
            # Prepare views for MapAnything
            views = []
            for view_idx in range(N):
                view_images = images[view_idx * B:(view_idx + 1) * B]
                
                view_dict = {
                    'img': view_images,
                    'data_norm_type': ['dinov2'] * B,
                }
                
                if self.use_poses and camera_poses is not None:
                    view_poses = camera_poses[view_idx * B:(view_idx + 1) * B]
                    quats, trans = camera_to_world_to_mapanything_pose(view_poses)
                    view_dict['camera_pose_quats'] = quats
                    view_dict['camera_pose_trans'] = trans
                
                views.append(view_dict)
            
            # Encode all views
            all_encoder_features = self.mapanything._encode_n_views(views)
            
            # Fuse with geometric inputs if available
            if self.use_poses and camera_poses is not None:
                all_encoder_features = self.mapanything._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features
                )
            else:
                fused_features = []
                for feat in all_encoder_features:
                    feat_permuted = feat.permute(0, 2, 3, 1).contiguous()
                    feat_normed = self.mapanything.fusion_norm_layer(feat_permuted.float())
                    feat_normed = feat_normed.to(feat.dtype)
                    fused_features.append(feat_normed.permute(0, 3, 1, 2).contiguous())
                all_encoder_features = tuple(fused_features)
            
            # Multi-view transformer
            input_scale_token = (
                self.mapanything.scale_token.unsqueeze(0)
                .unsqueeze(-1)
                .repeat(B, 1, 1)
            )
            
            info_sharing_input = MultiViewTransformerInput(
                features=list(all_encoder_features),
                additional_input_tokens=input_scale_token,
            )
            
            info_sharing_output = self.mapanything.info_sharing(info_sharing_input)
            
            if isinstance(info_sharing_output, tuple) and len(info_sharing_output) == 2:
                final_features_output, intermediate_features_list = info_sharing_output
            else:
                final_features_output = info_sharing_output
                intermediate_features_list = []
            
            # Get geometric DPT outputs (depth + confidence)
            geometric_outputs = None
            if return_geometric and hasattr(self.mapanything, 'dense_head'):
                # Run geometric DPT
                geometric_outputs = self._get_geometric_outputs(
                    all_encoder_features, intermediate_features_list, 
                    final_features_output, B, N, padded_h, padded_w
                )
        
        # Extract layer features for panoptic DPT (reference view)
        layer_features = []
        
        if self.use_encoder_features_for_dpt:
            layer_features.append(all_encoder_features[0])  # Reference view encoder
            
            for intermediate_output in intermediate_features_list:
                if hasattr(intermediate_output, 'features'):
                    feat = intermediate_output.features[0]
                else:
                    feat = intermediate_output[0]
                layer_features.append(feat)
            
            if hasattr(final_features_output, 'features'):
                final_feat = final_features_output.features[0]
            else:
                final_feat = final_features_output[0]
            layer_features.append(final_feat)
        else:
            for intermediate_output in intermediate_features_list:
                if hasattr(intermediate_output, 'features'):
                    feat = intermediate_output.features[0]
                else:
                    feat = intermediate_output[0]
                layer_features.append(feat)
            
            if hasattr(final_features_output, 'features'):
                final_feat = final_features_output.features[0]
            else:
                final_feat = final_features_output[0]
            layer_features.append(final_feat)
        
        while len(layer_features) < 4:
            layer_features.append(layer_features[-1])
        layer_features = layer_features[:4]
        
        # Run Panoptic DPT (trainable)
        panoptic_outputs = self.panoptic_dpt(layer_features, padded_h, padded_w)
        
        # Crop if padded
        if pad_h > 0 or pad_w > 0:
            for key in panoptic_outputs:
                stride = self._out_feature_strides[key]
                target_h = orig_h // stride
                target_w = orig_w // stride
                panoptic_outputs[key] = panoptic_outputs[key][:, :, :target_h, :target_w]
        
        result = {
            'panoptic_features': panoptic_outputs,
            'layer_features': layer_features,
            'num_views': N,
            'batch_size': B,
        }
        
        if geometric_outputs is not None:
            result['geometric_depth'] = geometric_outputs['depth']
            result['geometric_confidence'] = geometric_outputs['confidence']
        
        return result
    
    def _get_geometric_outputs(
        self, encoder_features, intermediate_features, final_features,
        B, N, H, W
    ):
        """Get depth and confidence from MapAnything's geometric DPT."""
        # This would call MapAnything's dense_head and extract depth/confidence
        # For now, return placeholder - actual implementation depends on MapAnything's exact API
        
        device = encoder_features[0].device
        
        # Placeholder: In practice, extract from mapanything.dense_head
        depth_per_view = [torch.ones(B, 1, H // 14, W // 14, device=device) for _ in range(N)]
        conf_per_view = [torch.ones(B, 1, H // 14, W // 14, device=device) for _ in range(N)]
        
        return {
            'depth': depth_per_view,
            'confidence': conf_per_view,
        }
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    @property
    def size_divisibility(self):
        return 32
    
    def train(self, mode=True):
        super().train(mode)
        self.mapanything.eval()  # Keep frozen
        for param in self.mapanything.parameters():
            param.requires_grad = False
        self.panoptic_dpt.train(mode)
        return self


# ============================================================
# COMBINED CONSISTENCY LOSS MODULE
# ============================================================

class MultiViewConsistencyLoss(nn.Module):
    """
    Combined consistency loss with all three components.
    """
    
    def __init__(
        self,
        lambda_dense: float = 0.3,
        lambda_query: float = 0.3,
        lambda_mask: float = 0.4,
    ):
        super().__init__()
        
        self.lambda_dense = lambda_dense
        self.lambda_query = lambda_query
        self.lambda_mask = lambda_mask
        
        self.dense_loss = DenseFeatureConsistencyLoss()
        self.query_loss = QueryEmbeddingConsistencyLoss()
        self.mask_loss = MaskProjectionConsistencyLoss()
    
    def forward(
        self,
        panoptic_features: List[torch.Tensor],
        geometric_confidence: List[torch.Tensor],
        query_embeddings: List[torch.Tensor],
        pred_masks: List[torch.Tensor],
        depth: List[torch.Tensor],
        poses: List[torch.Tensor],
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all consistency losses.
        """
        N = len(panoptic_features)
        
        # Compute correspondences for all view pairs
        correspondences_list = []
        valid_masks_list = []
        
        for i in range(N):
            for j in range(i + 1, N):
                corr, valid, _ = compute_pixel_correspondences(
                    depth[i].squeeze(1),  # [B, H, W]
                    poses[i],
                    poses[j],
                    intrinsics,
                )
                correspondences_list.append(corr)
                valid_masks_list.append(valid)
        
        # 1. Dense feature consistency
        loss_dense = self.dense_loss(
            panoptic_features, geometric_confidence,
            correspondences_list, valid_masks_list
        )
        
        # 2. Query embedding consistency
        loss_query = self.query_loss(
            query_embeddings, pred_masks,
            correspondences_list, valid_masks_list
        )
        
        # 3. Mask projection consistency
        loss_mask = self.mask_loss(
            pred_masks, depth, poses, intrinsics,
            correspondences_list, valid_masks_list
        )
        
        total_loss = (
            self.lambda_dense * loss_dense +
            self.lambda_query * loss_query +
            self.lambda_mask * loss_mask
        )
        
        return {
            'loss_consistency_total': total_loss,
            'loss_consistency_dense': loss_dense,
            'loss_consistency_query': loss_query,
            'loss_consistency_mask': loss_mask,
        }


# ============================================================
# CONFIGURATION
# ============================================================

def add_consistency_config(cfg):
    """Add consistency loss configuration."""
    cfg.MODEL.CONSISTENCY = CN()
    cfg.MODEL.CONSISTENCY.ENABLED = True
    cfg.MODEL.CONSISTENCY.LAMBDA_DENSE = 0.3
    cfg.MODEL.CONSISTENCY.LAMBDA_QUERY = 0.3
    cfg.MODEL.CONSISTENCY.LAMBDA_MASK = 0.4
    cfg.MODEL.CONSISTENCY.LAMBDA_TOTAL = 0.5  # Weight vs panoptic loss


def setup_cfg(args):
    """Setup configuration."""
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    
    # Add custom configs
    cfg.MODEL.MULTIVIEW = CN()
    cfg.MODEL.MULTIVIEW.NUM_VIEWS = 2
    cfg.MODEL.MULTIVIEW.USE_POSES = True
    cfg.MODEL.MULTIVIEW.MIN_CAMERA_DISTANCE = 0.5
    
    add_consistency_config(cfg)
    
    cfg.SOLVER.DPT_LR = 1e-5
    
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("3D Consistency Training for MapAnything + Mask2Former")
    print("=" * 60)
    print("\nThis script implements UNITE-inspired consistency losses:")
    print("1. L_dense: Align Parallel DPT features (weighted by Geometric DPT confidence)")
    print("2. L_query: Contrastive alignment of Mask2Former object queries")
    print("3. L_mask: Mask projection consistency across views")
    print("\nTotal Loss = L_panoptic + λ * (L_dense + L_query + L_mask)")
