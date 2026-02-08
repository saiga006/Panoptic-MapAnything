"""
Multi-View Training Script for MapAnything + Mask2Former on ScanNet++

This script extends the single-view training to support multi-view inputs
with camera poses for the MapAnything backbone, using Query Propagation
for cross-view consistency.

================================================================================
ARCHITECTURE OVERVIEW (Updated: January 2026)
================================================================================

Input Pipeline (ScanNet++ Dataset):
────────────────────────────────────
- ScanNetPPMultiViewDatasetMapper loads N views per scene
- Each view includes: RGB image, camera pose (4x4 c2w), camera intrinsics (3x3)
- Optional: Depth maps rendered from 3D mesh for spatial bridging
- Panoptic annotations: per-pixel segment IDs + category labels

Training Architecture:
─────────────────────

    Input: N Views [B, N, 3, H, W] + Camera Poses [B, N, 4, 4] + Intrinsics [B, N, 3, 3]
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  FROZEN: MapAnything Backbone (Multi-View Mode)                         │
    │  ├─ DINOv2 ViT-L Encoder (768-dim) - encodes all N views                │
    │  ├─ Optional Geometric Encoding (camera poses → tokens)                 │
    │  └─ Multi-View Info-Sharing Transformer (1024-dim cross-attention)      │
    │  Output: Shared multi-scale features for all N views                    │
    └─────────────────────────────────────────────────────────────────────────┘
        │ [B, N, C, H/14, W/14] features per view
        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRAINABLE: Panoptic DPT Head (per view)                                │
    │  ├─ Reassemble: ViT tokens → spatial features                          │
    │  ├─ Fusion blocks with skip connections                                 │
    │  └─ Output: res2 (1/4), res3 (1/8), res4 (1/16), res5 (1/32)           │
    │  Output: FPN-style features for Mask2Former                             │
    └─────────────────────────────────────────────────────────────────────────┘
        │ res2-res5 features for each view
        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRAINABLE: MSDeformAttn Pixel Decoder (SHARED across views)            │
    │  ├─ Multi-scale deformable attention                                    │
    │  └─ Output: mask_features [B, 256, H/4, W/4]                            │
    └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRAINABLE: Query Propagation Transformer Decoder (SHARED)              │
    │                                                                         │
    │  ═══════════════ REFERENCE VIEW (randomly selected) ═══════════════    │
    │  ├─ Initialize: 100 learnable object queries                            │
    │  ├─ 9-layer transformer with masked cross-attention                     │
    │  ├─ Each layer: cross-attn → self-attn → FFN → predict masks            │
    │  ├─ Output: refined queries Q_ref [100, B, 256]                         │
    │  └─ Supervision: Hungarian matching + CE + Dice + Mask BCE loss         │
    │                                                                         │
    │  ═══════════════════ TARGET VIEWS (N-1 views) ═══════════════════════  │
    │  For each target view:                                                  │
    │  ├─ Initialize: Q_ref from reference view (propagated queries)          │
    │  ├─ SPATIAL BRIDGING (if depth available):                              │
    │  │   ├─ Warp ref masks to target view using depth + poses               │
    │  │   └─ Alpha-blend: warped_mask * α + predicted_mask * (1-α)          │
    │  │       α = 1.0 (layer 0) → 0.0 (layer 9), linear decay               │
    │  ├─ 9-layer transformer with alpha-blended attention masks              │
    │  ├─ Output: target predictions with same query identity                 │
    │  └─ Supervision: Same as reference (queries maintain identity)          │
    │                                                                         │
    │  Key: Query #k predicts same semantic entity across ALL views           │
    └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LOSS COMPUTATION (NO KL DIVERGENCE)                                    │
    │  ├─ Reference View:                                                     │
    │  │   ├─ loss_ce: Cross-entropy classification loss                      │
    │  │   ├─ loss_mask: Binary mask loss                                     │
    │  │   └─ loss_dice: Dice loss for mask quality                           │
    │  ├─ Target Views (averaged):                                            │
    │  │   ├─ loss_ce_target: Average CE across target views                  │
    │  │   ├─ loss_mask_target: Average mask loss                             │
    │  │   └─ loss_dice_target: Average dice loss                             │
    │  └─ Total: ref_loss + avg(target_losses)                                │
    │                                                                         │
    │  Cross-view consistency is IMPLICIT via query propagation,              │
    │  NOT enforced by explicit KL divergence loss.                           │
    └─────────────────────────────────────────────────────────────────────────┘

Inference Architecture:
──────────────────────
- Process all N views through backbone (with info-sharing)
- Use view 0 as reference → get queries
- Optionally: aggregate predictions across views for 3D consistency
- Output: per-view panoptic predictions with consistent query IDs

Key Design Decisions:
────────────────────
1. SHARED decoder: Same weights for all views → unified semantic space
2. Query propagation: Reference queries used for all views → identity preservation
3. Alpha blending: Geometric guidance (warped masks) → learned refinement
4. No KL loss: Consistency is structural (same queries), not regularized
5. Warmup period: First K iterations train views independently

CSV Logging:
───────────
- training_metrics.csv: loss_ce, loss_mask, loss_dice, lr (every 20 iter)
- evaluation_metrics.csv: PQ, SQ, RQ, PQ_th, PQ_st (after each eval)
- view_analysis_metrics.csv: per-view breakdown (optional --view-analysis mode)

Configuration:
─────────────
- MODEL.MULTIVIEW.NUM_VIEWS: Number of views per scene (default: 8)
- MODEL.MULTIVIEW.WARMUP_ITER: Warmup iterations (default: 1000)
- MODEL.MULTIVIEW.USE_POSES: Enable camera pose processing (default: True)
- INPUT.LOAD_DEPTH: Load/render depth for spatial bridging (default: True)

================================================================================
"""

import os
import sys
import copy
import warnings
import logging
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Fix OpenCV threading issues in multiprocessing
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Use spawn method to avoid fork issues with CUDA/OpenGL
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Setup module logger
logger = logging.getLogger(__name__)

# Filter FutureWarnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# Detectron2 imports
from detectron2.layers import ShapeSpec
from detectron2.config import CfgNode as CN
from detectron2.engine import launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.modeling import Backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
from torch.utils.tensorboard import SummaryWriter

# Ensure local mask2former is importable
sys.path.insert(0, os.getcwd())

from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# MapAnything imports
from mapanything.models import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

# Import components from single-view training
from m2f_train_3d_loss import (
    PanopticDPTHead,
    NaNLossCheckHook,
    ResidualConvUnit,
    FeatureFusionBlock,
)


# ============================================================
# POSE CONVERSION UTILITIES
# ============================================================

def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (x, y, z, w).
    
    MapAnything expects quaternion in (x, y, z, w) format.
    
    Args:
        R: Rotation matrix [..., 3, 3]
    
    Returns:
        Quaternion [..., 4] in (x, y, z, w) format
    """
    # Adapted from pytorch3d
    batch_dim = R.shape[:-2]
    
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    
    trace = m00 + m11 + m22
    
    q = torch.zeros(*batch_dim, 4, device=R.device, dtype=R.dtype)
    
    # Case: trace > 0
    s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
    w = 0.25 * s
    x = (m21 - m12) / s
    y = (m02 - m20) / s
    z = (m10 - m01) / s
    
    # Return in (x, y, z, w) format
    q[..., 0] = x
    q[..., 1] = y
    q[..., 2] = z
    q[..., 3] = w
    
    return q


def camera_to_world_to_mapanything_pose(c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert camera-to-world 4x4 matrix to MapAnything pose format.
    
    Args:
        c2w: Camera-to-world transformation [B, 4, 4]
    
    Returns:
        quats: Quaternions [B, 4] in (x, y, z, w) format
        trans: Translations [B, 3]
    """
    R = c2w[:, :3, :3]  # [B, 3, 3]
    t = c2w[:, :3, 3]   # [B, 3]
    
    quats = matrix_to_quaternion(R)  # [B, 4]
    
    return quats, t


# ============================================================
# GEOMETRIC WARPING FOR ATTENTION MASK BRIDGING
# ============================================================

def warp_masks_to_target_view(
    masks: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    tgt_depth: Optional[torch.Tensor] = None,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Warp attention masks from source (reference) view to target view.
    
    Strategies:
    1. If tgt_depth is provided (BEST): Use Backward Warping (Target -> Source)
       - Unproject target pixels using target depth -> Transform to Source -> Sample Source masks
       - Dense, accurate, differentiable, no holes.
       
    2. If only ref_depth is provided (BACKUP): Use Forward Splatting (Source -> Target)
       - Unproject source pixels using source depth -> Transform to Target -> Splat
       - Can have holes/cracks, uses scatter_add approximation.
    
    Args:
        masks: Source view masks [B, Q, H, W] - probabilities (0-1)
        ref_depth: Source view depth [B, 1, H, W] in meters
        ref_pose: Source camera-to-world [B, 4, 4]
        tgt_pose: Target camera-to-world [B, 4, 4]
        intrinsics: Camera intrinsics [B, 3, 3] (same for both views)
        tgt_depth: Optional Target view depth [B, 1, H, W] for backward warping
        mask_threshold: Threshold for binary mask
    
    Returns:
        Warped masks [B, Q, H, W] in target view coordinates
    """
    B, Q, H, W = masks.shape
    device = masks.device
    dtype = masks.dtype
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: Backward Warping (Preferred) - Requires Target Depth
    # -------------------------------------------------------------------------
    if tgt_depth is not None:
        # Resize target depth to mask resolution if needed
        if tgt_depth.shape[-2:] != (H, W):
            tgt_depth = F.interpolate(tgt_depth, size=(H, W), mode='nearest')
            
        depth_vals = tgt_depth.squeeze(1) # [B, H, W]
        
        # 1. Create pixel grid for TARGET view
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # 2. Unproject Target -> 3D Target Frame
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)
        
        x_norm = (pixel_coords[..., 0] - cx) / fx
        y_norm = (pixel_coords[..., 1] - cy) / fy
        
        pts_tgt_cam = torch.stack([x_norm * depth_vals, y_norm * depth_vals, depth_vals], dim=-1) # [B, H, W, 3]
        
        # 3. Transform 3D Target -> World -> Reference Frame
        # pts_world = T_tgt @ pts_tgt
        pts_tgt_cam_h = torch.cat([pts_tgt_cam, torch.ones_like(pts_tgt_cam[..., :1])], dim=-1) # [B, H, W, 4]
        pts_world = torch.einsum('bhwi,bji->bhwj', pts_tgt_cam_h, tgt_pose)
        
        # pts_ref = T_ref_inv @ pts_world
        ref_pose_inv = torch.inverse(ref_pose)
        pts_ref_cam = torch.einsum('bhwi,bji->bhwj', pts_world, ref_pose_inv)
        pts_ref_cam = pts_ref_cam[..., :3]
        
        # 4. Project Reference Frame -> Reference Pixels
        z_ref = pts_ref_cam[..., 2].clamp(min=1e-6)
        x_ref = pts_ref_cam[..., 0] / z_ref
        y_ref = pts_ref_cam[..., 1] / z_ref
        
        u_ref = x_ref * fx + cx
        v_ref = y_ref * fy + cy
        
        # 5. Normalize for grid_sample ([-1, 1])
        u_norm = 2.0 * u_ref / (W - 1) - 1.0
        v_norm = 2.0 * v_ref / (H - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1) # [B, H, W, 2]
        
        # 6. Sample using grid_sample (Bilinear Interpolation)
        # We sample FROM the Reference Masks, using the calculated Reference Coordinates
        warped_masks = F.grid_sample(
            masks.float(), 
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Set invalid pixels (behind camera etc) to 0
        valid_mask = (pts_ref_cam[..., 2] > 0) & (depth_vals > 0)
        warped_masks = warped_masks * valid_mask.unsqueeze(1).float()
        
        return warped_masks

    # -------------------------------------------------------------------------
    # STRATEGY 2: Forward Splatting (Backup) - Only Ref Depth
    # -------------------------------------------------------------------------
    
    # Create pixel grid for source view
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    # [H, W, 2] -> [B, H, W, 2]
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Get depth values [B, H, W]
    depth_vals = ref_depth.squeeze(1)
    
    # Unproject to 3D: x_cam = K^{-1} @ [u, v, 1]^T * depth
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)
    
    # Pixel to normalized camera coordinates
    x_norm = (pixel_coords[..., 0] - cx) / fx  # [B, H, W]
    y_norm = (pixel_coords[..., 1] - cy) / fy  # [B, H, W]
    
    # 3D points in source camera frame [B, H, W, 3]
    pts_src_cam = torch.stack([
        x_norm * depth_vals,
        y_norm * depth_vals,
        depth_vals
    ], dim=-1)
    
    # Transform to world coordinates: X_world = T_src @ X_src
    pts_src_cam_h = torch.cat([
        pts_src_cam,
        torch.ones(B, H, W, 1, device=device, dtype=dtype)
    ], dim=-1)  # [B, H, W, 4]
    
    # [B, H, W, 4] @ [B, 4, 4]^T -> [B, H, W, 4]
    pts_world = torch.einsum('bhwi,bji->bhwj', pts_src_cam_h, src_pose)
    
    # Transform to target camera: X_tgt = T_tgt^{-1} @ X_world
    tgt_pose_inv = torch.inverse(tgt_pose)
    pts_tgt_cam = torch.einsum('bhwi,bji->bhwj', pts_world, tgt_pose_inv)
    pts_tgt_cam = pts_tgt_cam[..., :3]  # [B, H, W, 3]
    
    # Project to target image plane
    z_tgt = pts_tgt_cam[..., 2].clamp(min=1e-6)  # [B, H, W]
    x_tgt = pts_tgt_cam[..., 0] / z_tgt  # normalized
    y_tgt = pts_tgt_cam[..., 1] / z_tgt
    
    # To pixel coordinates
    u_tgt = x_tgt * fx + cx  # [B, H, W]
    v_tgt = y_tgt * fy + cy
    
    # Round to integer coordinates
    u_base = torch.round(u_tgt).long()
    v_base = torch.round(v_tgt).long()
    
    # Valid mask: points that project within image bounds and have positive depth
    valid = (
        (u_base >= 0) & (u_base < W) &
        (v_base >= 0) & (v_base < H) &
        (pts_tgt_cam[..., 2] > 0) &
        (depth_vals > 0)
    )  # [B, H, W]
    
    # --- Forward Splatting using Scatter Add ---
    
    # 1. Permute masks to [B, H, W, Q] for easier indexing
    masks_perm = masks.permute(0, 2, 3, 1)
    
    # 2. Extract valid values and indices
    if not valid.any():
        # Edge case: No valid projections
        return torch.zeros(B, Q, H, W, device=device, dtype=dtype)
        
    src_vals = masks_perm[valid]  # [N_valid, Q]
    
    # 3. Calculate 1D indices for scattering
    # We scatter into a flattened tensor of shape [B*H*W, Q]
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    flat_b = batch_indices[valid]
    flat_v = v_base[valid]
    flat_u = u_base[valid]
    
    # Index in flattened [B, H, W] array
    flat_tgt_indices = flat_b * (H * W) + flat_v * W + flat_u # [N_valid]
    
    # 4. Filter out collisions (optional, max would be better but add is faster/easier)
    # With add, we accumulate probabilities. We can clamp later.
    
    # output_flat: [B*H*W, Q]
    output_flat = torch.zeros(B * H * W, Q, device=device, dtype=dtype)
    
    # Scatter add
    # indices needs to be broadcastable to src_vals
    # output.scatter_add_(dim, index, src)
    # We treat dim 0 as the spatial dimension to scatter into
    indices_expanded = flat_tgt_indices.unsqueeze(1).expand(-1, Q) # [N_valid, Q]
    
    output_flat.scatter_add_(0, indices_expanded, src_vals)
    
    # 5. Reshape and Clamp
    # [B*H*W, Q] -> [B, H, W, Q] -> [B, Q, H, W]
    warped_masks = output_flat.view(B, H, W, Q).permute(0, 3, 1, 2)
    
    # Clamp to [0, 1] since we added probabilities
    warped_masks.clamp_(max=1.0)
    
    return warped_masks


def create_warped_attention_mask(
    ref_pred_masks: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    target_size: Tuple[int, int],
    tgt_depth: Optional[torch.Tensor] = None,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Create attention masks for target view decoder by warping reference predictions.
    """
    # Downsample depth if needed (to match low-res masks)
    H_mask, W_mask = ref_pred_masks.shape[-2:]
    H_depth, W_depth = ref_depth.shape[-2:]
    
    # Scale intrinsics if needed (common parsing)
    if H_mask != H_depth:
         # Scale intrinsics logic repeated for safety
         scale_x = W_mask / W_depth
         scale_y = H_mask / H_depth
         intrinsics = intrinsics.clone()
         intrinsics[:, 0, 0] *= scale_x
         intrinsics[:, 1, 1] *= scale_y
         intrinsics[:, 0, 2] *= scale_x
         intrinsics[:, 1, 2] *= scale_y

         if H_mask < H_depth:
             ref_depth = F.interpolate(ref_depth, size=(H_mask, W_mask), mode='nearest')
    
    # Warp masks to target view (Backend handles Forward/Backward choice)
    warped_masks = warp_masks_to_target_view(
        masks=ref_pred_masks.sigmoid(),
        ref_depth=ref_depth,
        ref_pose=ref_pose,
        tgt_pose=tgt_pose,
        intrinsics=intrinsics,
        tgt_depth=tgt_depth, # Pass target depth if available
        mask_threshold=mask_threshold,
    )
    
    # Resize to target size for attention
    if warped_masks.shape[-2:] != target_size:
        warped_masks = F.interpolate(
            warped_masks,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
    
    # Convert to attention mask format
    # Mask2Former uses: True = ignore, False = attend
    # We want to attend where the mask is present
    attn_mask = warped_masks < mask_threshold  # [B, Q, H', W']
    
    # Safety Check: If a query has NO valid attention regions (all True/Ignore),
    # this will cause NaN in Softmax. 
    # Fallback to attending everywhere (all False) to allow the model to search.
    # Check per query: [B, Q]
    all_ignore = attn_mask.flatten(2).all(dim=2)  # [B, Q]
    
    if all_ignore.any():
        # Create a broadcastable mask for indexing
        # Expand all_ignore to [B, Q, 1, 1] then to [B, Q, H', W']
        mask_to_reset = all_ignore.unsqueeze(-1).unsqueeze(-1).expand_as(attn_mask)
        attn_mask[mask_to_reset] = False # Set to Attend Everywhere
    
    return attn_mask


# ============================================================
# MULTI-VIEW BACKBONE
# ============================================================

class MapAnythingMultiViewBackbone(Backbone):
    """
    Multi-View MapAnything Backbone for Query Propagation.
    
    Key features:
    1. Accepts multiple views with camera poses
    2. Uses MapAnything's full multi-view processing pipeline
    3. Panoptic DPT (trainable): outputs res2-res5 features for segmentation
    4. Returns features for ALL views to enable query propagation
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        self.patch_size = 14  # DINOv2 patch size
        self.encoder_dim = 768  # DINOv2 ViT-L encoder dim
        self.info_sharing_dim = 1024  # After info_sharing transformer
        
        # Number of views to process simultaneously
        self.num_views = cfg.MODEL.MULTIVIEW.NUM_VIEWS
        self.use_poses = cfg.MODEL.MULTIVIEW.USE_POSES
        
        # 3D Consistency settings
        # Load MapAnything model
        print("Loading MapAnything model...")
        
        # Get checkpoint path from config (set via MODEL.WEIGHTS or MODEL.MAPANYTHING.CHECKPOINT_PATH)
        if hasattr(cfg.MODEL, 'MAPANYTHING') and hasattr(cfg.MODEL.MAPANYTHING, 'CHECKPOINT_PATH'):
            mapanything_path = cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH
        else:
            # Fallback to MODEL.WEIGHTS if MAPANYTHING config not set
            mapanything_path = cfg.MODEL.WEIGHTS if hasattr(cfg.MODEL, 'WEIGHTS') else None
        
        if not mapanything_path:
            raise ValueError(
                "MapAnything checkpoint path not specified. "
                "Set MODEL.MAPANYTHING.CHECKPOINT_PATH or MODEL.WEIGHTS in config."
            )
        
        if not os.path.exists(mapanything_path):
            raise FileNotFoundError(f"MapAnything checkpoint not found: {mapanything_path}")
        
        print(f"Loading MapAnything from: {mapanything_path}")
        
        self.mapanything = MapAnything.from_pretrained(
            mapanything_path,
            local_files_only=True
        )
        
        # Ensure model is consistently in float32 to avoid mixed-precision parameter issues
        # (e.g. weight=float32, bias=float16) which crash conv2d even with autocast
        self.mapanything.float()
        
        # The DINOv2 encoder is loaded from torch hub inside MapAnything.encoder.model
        # and may have mixed precision. Force it to float32 explicitly.
        if hasattr(self.mapanything, 'encoder') and hasattr(self.mapanything.encoder, 'model'):
            print("Converting DINOv2 encoder to float32...")
            self.mapanything.encoder.model.float()
            
            # Also check for nested modules like patch_embed
            for name, module in self.mapanything.encoder.model.named_modules():
                # Convert all parameters
                for pname, param in module.named_parameters(recurse=False):
                    if param.dtype == torch.float16:
                        print(f"  Converting param {name}.{pname} from Half to float32")
                        param.data = param.data.to(torch.float32)
                # Convert all buffers  
                for bname, buf in module.named_buffers(recurse=False):
                    if buf is not None and buf.dtype == torch.float16:
                        print(f"  Converting buffer {name}.{bname} from Half to float32")
                        buf.data = buf.data.to(torch.float32)
        
        # Paranoid check: Force all parameters and buffers to float32 explicitly
        print("Verifying MapAnything parameter and buffer types...")
        fixed_params = 0
        fixed_buffers = 0
        for name, param in self.mapanything.named_parameters():
            if param.dtype == torch.float16:
                print(f"  WARNING: Found Half parameter {name}. Casting to float32.")
                param.data = param.data.to(torch.float32)
                fixed_params += 1
        for name, buf in self.mapanything.named_buffers():
            if buf is not None and buf.dtype == torch.float16:
                print(f"  WARNING: Found Half buffer {name}. Casting to float32.")
                buf.data = buf.data.to(torch.float32)
                fixed_buffers += 1
                
        if fixed_params > 0 or fixed_buffers > 0:
            print(f"  Fixed {fixed_params} parameters and {fixed_buffers} buffers to float32.")
        else:
            print("  All parameters and buffers are float32.")

        # Update dimensions from loaded model to ensure correctness
        if hasattr(self.mapanything, 'encoder'):
            # Try to find embedding dimension
            if hasattr(self.mapanything.encoder, 'enc_embed_dim'):
                self.encoder_dim = self.mapanything.encoder.enc_embed_dim
            elif hasattr(self.mapanything.encoder, 'embed_dim'):
                self.encoder_dim = self.mapanything.encoder.embed_dim
            elif hasattr(self.mapanything.encoder, 'model') and hasattr(self.mapanything.encoder.model, 'embed_dim'):
                self.encoder_dim = self.mapanything.encoder.model.embed_dim
            
            # Try to find patch size
            if hasattr(self.mapanything.encoder, 'patch_size'):
                self.patch_size = self.mapanything.encoder.patch_size
            elif hasattr(self.mapanything.encoder, 'model') and hasattr(self.mapanything.encoder.model, 'patch_size'):
                self.patch_size = self.mapanything.encoder.model.patch_size

        if hasattr(self.mapanything, 'info_sharing'):
            if hasattr(self.mapanything.info_sharing, 'dim'):
                self.info_sharing_dim = self.mapanything.info_sharing.dim
        
        print(f"Detected MapAnything config: Encoder Dim={self.encoder_dim}, InfoSharing Dim={self.info_sharing_dim}, Patch Size={self.patch_size}")

        self.mapanything.eval()
        
        for param in self.mapanything.parameters():
            param.requires_grad = False
        
        print(f"MapAnything loaded. Multi-view mode with {self.num_views} views")
        print(f"Using camera poses: {self.use_poses}")
        
        # Determine input dims for Panoptic DPT
        if hasattr(self.mapanything, 'use_encoder_features_for_dpt'):
            self.use_encoder_features_for_dpt = self.mapanything.use_encoder_features_for_dpt
        else:
            self.use_encoder_features_for_dpt = True
        
        if self.use_encoder_features_for_dpt:
            input_dims = [self.encoder_dim] + [self.info_sharing_dim] * 3
        else:
            input_dims = [self.info_sharing_dim] * 4
        
        # Initialize Panoptic DPT head (TRAINABLE)
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
        
        # Store intermediate features for multi-view consistency
        self._all_view_layer_features = None
    
    def _init_panoptic_dpt_from_geometric(self):
        """Initialize Panoptic DPT weights from MapAnything's geometric DPT."""
        # Same as single-view version
        print("\nInitializing Panoptic DPT from Geometric DPT...")
        
        geometric_dpt = None
        if hasattr(self.mapanything, 'dpt_feature_head'):
            geometric_dpt = self.mapanything.dpt_feature_head
        elif hasattr(self.mapanything, 'dense_head') and len(self.mapanything.dense_head) > 0:
            geometric_dpt = self.mapanything.dense_head[0]
        
        if geometric_dpt is None:
            print("  WARNING: Could not find geometric DPT head in MapAnything")
            return
        
        panoptic_dpt = self.panoptic_dpt
        copied_count = 0
        
        def copy_weights(src, dst, name):
            nonlocal copied_count
            try:
                if hasattr(src, 'weight') and hasattr(dst, 'weight'):
                    if src.weight.shape == dst.weight.shape:
                        dst.weight.data.copy_(src.weight.data)
                        copied_count += 1
                if hasattr(src, 'bias') and hasattr(dst, 'bias'):
                    if src.bias is not None and dst.bias is not None:
                        if src.bias.shape == dst.bias.shape:
                            dst.bias.data.copy_(src.bias.data)
                return True
            except Exception as e:
                return False
        
        # Copy matching weights from geometric DPT
        if hasattr(geometric_dpt, 'resize_layers'):
            for i in range(min(len(geometric_dpt.resize_layers), len(panoptic_dpt.resize_layers))):
                if not isinstance(geometric_dpt.resize_layers[i], nn.Identity):
                    copy_weights(geometric_dpt.resize_layers[i], panoptic_dpt.resize_layers[i], f"resize_layers[{i}]")
        
        if hasattr(geometric_dpt, 'scratch'):
            for layer_name in ['layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn']:
                if hasattr(geometric_dpt.scratch, layer_name) and hasattr(panoptic_dpt.scratch, layer_name):
                    copy_weights(getattr(geometric_dpt.scratch, layer_name),
                               getattr(panoptic_dpt.scratch, layer_name), f"scratch.{layer_name}")
            
            for refinenet_name in ['refinenet1', 'refinenet2', 'refinenet3', 'refinenet4']:
                if hasattr(geometric_dpt.scratch, refinenet_name) and hasattr(panoptic_dpt.scratch, refinenet_name):
                    src_block = getattr(geometric_dpt.scratch, refinenet_name)
                    dst_block = getattr(panoptic_dpt.scratch, refinenet_name)
                    
                    if hasattr(src_block, 'out_conv'):
                        copy_weights(src_block.out_conv, dst_block.out_conv, f"scratch.{refinenet_name}.out_conv")
                    
                    for rcu in ['resConfUnit1', 'resConfUnit2']:
                        if hasattr(src_block, rcu) and hasattr(dst_block, rcu):
                            for conv_name in ['conv1', 'conv2']:
                                if hasattr(getattr(src_block, rcu), conv_name):
                                    copy_weights(getattr(getattr(src_block, rcu), conv_name),
                                               getattr(getattr(dst_block, rcu), conv_name),
                                               f"scratch.{refinenet_name}.{rcu}.{conv_name}")
        
        print(f"  ✓ Copied {copied_count} weight tensors from geometric DPT")
    
    def forward(
        self,
        images: torch.Tensor,
        camera_poses: Optional[torch.Tensor] = None,
        camera_intrinsics: Optional[torch.Tensor] = None,
        return_all_views: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-view forward pass through MapAnything backbone and Panoptic DPT.
        
        Args:
            images: Input images [B, N, 3, H, W] where N is num_views
                    OR [B*N, 3, H, W] if already flattened
            camera_poses: Camera-to-world transforms [B, N, 4, 4]
                         OR [B*N, 4, 4] if flattened
            camera_intrinsics: Camera intrinsics [B, N, 3, 3] (optional)
            return_all_views: If True, return features for all N views (default: True)
        
        Returns:
            Dictionary with:
            - res2, res3, res4, res5: Panoptic features for view 0 (for backwards compat)
            - all_view_features: Dict mapping feature names to [N, B, C, H', W'] tensors
        """
        # Handle input shape
        if images.dim() == 5:
            # [B, N, 3, H, W] -> [B*N, 3, H, W]
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W)
            if camera_poses is not None and camera_poses.dim() == 4:
                camera_poses = camera_poses.view(B * N, 4, 4)
        else:
            B_N, C, H, W = images.shape
            N = self.num_views
            B = B_N // N
            images_flat = images
        
        orig_h, orig_w = H, W
        
        # Pad to be divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            images_flat = F.pad(images_flat, (0, pad_w, 0, pad_h))
        
        padded_h, padded_w = images_flat.shape[2], images_flat.shape[3]
        
        # Store geometric outputs if requested
        geometric_outputs = {}
        
        # CRITICAL: Disable autocast for the frozen backbone
        # We dynamically match the input dtype to the model's actual weight dtype
        # to prevent "Input type (float) and bias type (c10::Half)" mismatches.
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            # Determine target dtype from the model itself (specifically the DINOv2 entry conv layer)
            target_dtype = torch.float32
            try:
                # Target the specific DINOv2 conv layer known to fail
                if hasattr(self.mapanything, 'encoder') and \
                   hasattr(self.mapanything.encoder, 'model') and \
                   hasattr(self.mapanything.encoder.model, 'patch_embed') and \
                   hasattr(self.mapanything.encoder.model.patch_embed, 'proj'):
                    target_dtype = self.mapanything.encoder.model.patch_embed.proj.weight.dtype
            except Exception:
                # Fallback to first parameter of the model
                try:
                    target_dtype = next(self.mapanything.parameters()).dtype
                except:
                    pass
            
            # Cast inputs to match model weights (whether they remained Float32 or became Half)
            images_flat = images_flat.to(dtype=target_dtype)
            
            # Prepare views for MapAnything
            views = []
            for view_idx in range(N):
                view_images = images_flat[view_idx::N]  # Select every N-th starting from view_idx
                
                view_dict = {
                    'img': view_images,
                    'data_norm_type': ['dinov2'] * (B),
                }
                
                # Add camera poses if available
                if self.use_poses and camera_poses is not None:
                    view_poses = camera_poses[view_idx::N]  # [B, 4, 4]
                    quats, trans = camera_to_world_to_mapanything_pose(view_poses)
                    view_dict['camera_pose_quats'] = quats  # [B, 4]
                    view_dict['camera_pose_trans'] = trans  # [B, 3]
                
                views.append(view_dict)
            
            # Encode all views through DINOv2
            all_encoder_features = self.mapanything._encode_n_views(views)
            # Result: tuple of N tensors, each [B, C, patch_h, patch_w]
            
            # Fuse with optional geometric inputs if poses provided
            if self.use_poses and camera_poses is not None:
                all_encoder_features = self.mapanything._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features
                )
            else:
                # Just normalize encoder features
                fused_features = []
                for feat in all_encoder_features:
                    feat_permuted = feat.permute(0, 2, 3, 1).contiguous()
                    feat_normed = self.mapanything.fusion_norm_layer(feat_permuted.float())
                    feat_normed = feat_normed.to(feat.dtype)
                    fused_features.append(feat_normed.permute(0, 3, 1, 2).contiguous())
                all_encoder_features = tuple(fused_features)
            
            # Prepare scale token
            input_scale_token = (
                self.mapanything.scale_token.unsqueeze(0)
                .unsqueeze(-1)
                .repeat(B, 1, 1)
            )
            
            # Multi-view transformer input
            info_sharing_input = MultiViewTransformerInput(
                features=list(all_encoder_features),  # List of N tensors
                additional_input_tokens=input_scale_token,
            )
            
            # Run multi-view attention
            info_sharing_output = self.mapanything.info_sharing(info_sharing_input)
            
            if isinstance(info_sharing_output, tuple) and len(info_sharing_output) == 2:
                final_features_output, intermediate_features_list = info_sharing_output
            else:
                final_features_output = info_sharing_output
                intermediate_features_list = []
        
        # Process ALL views through Panoptic DPT for query propagation
        all_view_panoptic_features = {}
        for view_idx in range(N):
            view_layer_features = self._extract_layer_features_for_view(
                view_idx, all_encoder_features, intermediate_features_list, final_features_output
            )
            view_outputs = self.panoptic_dpt(view_layer_features, padded_h, padded_w)
            
            for key, val in view_outputs.items():
                if key not in all_view_panoptic_features:
                    all_view_panoptic_features[key] = []
                all_view_panoptic_features[key].append(val)
        
        # Stack per-view features: [N, B, C, H', W']
        for key in all_view_panoptic_features:
            all_view_panoptic_features[key] = torch.stack(all_view_panoptic_features[key], dim=0)
        
        # Crop if padded
        if pad_h > 0 or pad_w > 0:
            for key in all_view_panoptic_features:
                stride = self._out_feature_strides[key]
                target_h = orig_h // stride
                target_w = orig_w // stride
                all_view_panoptic_features[key] = all_view_panoptic_features[key][:, :, :, :target_h, :target_w]
        
        # Store reference (for backwards compatibility)
        self._all_view_layer_features = all_view_panoptic_features
        
        # Also return reference view features at top level for backwards compat
        outputs = {}
        for key in ['res2', 'res3', 'res4', 'res5']:
            outputs[key] = all_view_panoptic_features[key][0]  # View 0 features [B, C, H', W']
        
        outputs['all_view_features'] = all_view_panoptic_features
        outputs['num_views'] = N
        outputs['batch_size'] = B
        
        return outputs
    
    def _extract_layer_features_for_view(
        self,
        view_idx: int,
        all_encoder_features: tuple,
        intermediate_features_list: list,
        final_features_output,
    ) -> List[torch.Tensor]:
        """Extract 4 layer features for a specific view."""
        layer_features = []
        
        if self.use_encoder_features_for_dpt:
            # Layer 0: Encoder features from this view
            layer_features.append(all_encoder_features[view_idx])
            
            # Layers 1-2: Intermediate features
            for intermediate_output in intermediate_features_list:
                if hasattr(intermediate_output, 'features'):
                    feat = intermediate_output.features[view_idx]
                else:
                    feat = intermediate_output[view_idx]
                layer_features.append(feat)
            
            # Layer 3: Final features
            if hasattr(final_features_output, 'features'):
                final_feat = final_features_output.features[view_idx]
            else:
                final_feat = final_features_output[view_idx]
            layer_features.append(final_feat)
        else:
            for intermediate_output in intermediate_features_list:
                if hasattr(intermediate_output, 'features'):
                    feat = intermediate_output.features[view_idx]
                else:
                    feat = intermediate_output[view_idx]
                layer_features.append(feat)
            
            if hasattr(final_features_output, 'features'):
                final_feat = final_features_output.features[view_idx]
            else:
                final_feat = final_features_output[view_idx]
            layer_features.append(final_feat)
        
        # Ensure 4 layers
        while len(layer_features) < 4:
            layer_features.append(layer_features[-1])
        layer_features = layer_features[:4]
        
        return layer_features
    
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
        self.mapanything.eval()
        for param in self.mapanything.parameters():
            param.requires_grad = False
        self.panoptic_dpt.train(mode)
        return self


# ============================================================
# MULTI-VIEW DATA COLLATOR
# ============================================================

@dataclass
class MultiViewBatch:
    """Container for multi-view batched data."""
    images: torch.Tensor          # [B, N, 3, H, W]
    camera_poses: torch.Tensor    # [B, N, 4, 4]
    camera_intrinsics: torch.Tensor  # [B, N, 3, 3]
    sem_seg: torch.Tensor         # [B, N, H, W] semantic segmentation
    instances: List[Any]          # [B * N] Instances objects
    file_names: List[str]         # [B * N] file names
    scene_ids: List[str]          # [B] scene identifiers


def multi_view_collate_fn(batch_list: List[Dict]) -> Dict:
    """
    Custom collate function for multi-view batches.
    
    Input format from dataset mapper (list of dicts):
    [
        {
            'scene_id': str,
            'views': [
                {
                    'image': Tensor [3, H, W],
                    'camera_pose': Tensor [4, 4],
                    'camera_intrinsic': Tensor [3, 3],
                    'sem_seg': Tensor [H, W],
                    'instances': Instances,
                    'file_name': str,
                },
                ... (num_views per scene)
            ]
        },
        ...
    ]
    
    Output format for model:
    {
        'images': Tensor [B, N, 3, H, W],
        'camera_poses': Tensor [B, N, 4, 4],
        'camera_intrinsics': Tensor [B, N, 3, 3],
        'sem_seg': Tensor [B, N, H, W],
        'instances': List[Instances] (B * N),
        'file_names': List[str] (B * N),
        'scene_ids': List[str] (B),
    }
    """
    B = len(batch_list)
    N = len(batch_list[0]['views'])
    
    # Get channel count from first view
    first_view = batch_list[0]['views'][0]
    C = first_view['image'].shape[0]
    
    # Find maximum H, W across ALL views in ALL samples in the batch
    # Different scenes may have different augmentation sizes
    max_H = 0
    max_W = 0
    for sample in batch_list:
        for view in sample['views']:
            _, h, w = view['image'].shape
            max_H = max(max_H, h)
            max_W = max(max_W, w)
    
    # Apply size divisibility (typically 32 for Mask2Former)
    size_div = 32
    max_H = int(np.ceil(max_H / size_div) * size_div)
    max_W = int(np.ceil(max_W / size_div) * size_div)
    
    H, W = max_H, max_W
    
    # Check if depth is available
    has_depth = 'depth' in first_view and first_view['depth'] is not None
    
    # Allocate tensors with padded dimensions
    images = torch.zeros(B, N, C, H, W)
    camera_poses = torch.zeros(B, N, 4, 4)
    camera_intrinsics = torch.zeros(B, N, 3, 3)
    sem_seg = torch.zeros(B, N, H, W, dtype=torch.long)
    
    # Depth tensor (optional)
    if has_depth:
        depth = torch.zeros(B, N, 1, H, W)
    else:
        depth = None
    
    instances_list = []
    file_names_list = []
    scene_ids_list = []
    
    for b_idx, sample in enumerate(batch_list):
        scene_ids_list.append(sample['scene_id'])
        
        for v_idx, view in enumerate(sample['views']):
            # Get actual image dimensions
            img = view['image']
            _, h, w = img.shape
            
            # Copy image to padded tensor (top-left aligned)
            images[b_idx, v_idx, :, :h, :w] = img
            camera_poses[b_idx, v_idx] = view['camera_pose']
            camera_intrinsics[b_idx, v_idx] = view['camera_intrinsic']
            
            if 'sem_seg' in view:
                seg = view['sem_seg']
                sem_seg[b_idx, v_idx, :seg.shape[0], :seg.shape[1]] = seg
            
            if has_depth and 'depth' in view and view['depth'] is not None:
                d = view['depth']
                depth[b_idx, v_idx, :, :d.shape[1], :d.shape[2]] = d
            
            instances_list.append(view.get('instances', None))
            file_names_list.append(view.get('file_name', ''))
    
    result = {
        'images': images,
        'camera_poses': camera_poses,
        'camera_intrinsics': camera_intrinsics,
        'sem_seg': sem_seg,
        'instances': instances_list,
        'file_names': file_names_list,
        'scene_ids': scene_ids_list,
    }
    
    if depth is not None:
        result['depth'] = depth
    
    return result


# ============================================================
# MULTI-VIEW PANOPTIC MODEL
# ============================================================

class QueryPropagationTransformerDecoder(nn.Module):
    """
    Wrapper around Mask2Former's transformer decoder that supports query propagation.
    
    This allows passing refined queries from a reference view to target views,
    along with warped attention masks for spatial bridging.
    
    Key insight: When propagating queries to target view, we need to tell the
    decoder WHERE to look in the new image. This is done via initial_attn_mask,
    which is the reference view's masks warped to target view coordinates.
    
    ALPHA BLENDING STRATEGY:
    - Early layers: Use warped geometric mask (alpha ≈ 1.0)
    - Later layers: Gradually transition to self-predicted masks (alpha → 0.0)
    - This provides smooth transition from geometric guidance to learned refinement
    """
    
    def __init__(self, original_decoder: nn.Module):
        super().__init__()
        self.decoder = original_decoder
    
    def _prepare_warped_mask_for_layer(
        self,
        initial_attn_mask: torch.Tensor,
        target_size: Tuple[int, int],
        num_heads: int,
        bs: int,
    ) -> torch.Tensor:
        """
        Prepare warped attention mask for a specific layer's spatial size.
        
        Args:
            initial_attn_mask: Warped mask [B, Q, H_orig, W_orig]
            target_size: Target (H', W') for this layer
            num_heads: Number of attention heads
            bs: Batch size
        
        Returns:
            Warped mask in attention format [B * num_heads, Q, H' * W']
        """
        target_H, target_W = target_size
        
        # Resize to match target size
        warped_mask = F.interpolate(
            initial_attn_mask.float(),
            size=(target_H, target_W),
            mode='bilinear',
            align_corners=False,
        )  # [B, Q, H', W']
        
        # Flatten spatial dims: [B, Q, H' * W']
        warped_mask_flat = warped_mask.flatten(2)
        
        # Expand for multi-head attention: [B * num_heads, Q, H' * W']
        warped_attn_mask = warped_mask_flat.unsqueeze(1).repeat(1, num_heads, 1, 1)
        warped_attn_mask = warped_attn_mask.flatten(0, 1)  # [B*num_heads, Q, H'*W']
        
        return warped_attn_mask
    
    def _blend_attention_masks(
        self,
        warped_mask: torch.Tensor,
        predicted_mask: torch.Tensor,
        alpha: float,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Blend warped geometric mask with self-predicted mask.
        
        Args:
            warped_mask: Geometric warped mask [B*H, Q, HW] (float, 0-1 soft mask)
            predicted_mask: Self-predicted mask [B*H, Q, HW] (bool)
            alpha: Blending weight (1.0 = all warped, 0.0 = all predicted)
            threshold: Threshold for converting blended mask to bool
        
        Returns:
            Blended attention mask [B*H, Q, HW] (bool)
        """
        # Memory optimization: perform blending in-place if possible
        # warped_mask is float, predicted_mask is typically bool or float
        
        # Convert predicted mask to float if needed (but avoid copy if possible)
        if predicted_mask.dtype == torch.bool:
            predicted_float = predicted_mask.float()
        else:
            predicted_float = predicted_mask
            
        # Alpha blend: alpha * warped + (1 - alpha) * predicted
        # Use addcmul or similar to reduce peak memory
        # blended = alpha * warped_mask + (1.0 - alpha) * predicted_float
        
        # In-place blending to potential save memory
        warped_mask.mul_(alpha)
        warped_mask.add_(predicted_float, alpha=1.0 - alpha)
        # This modifies input warped_mask, which is okay if it's discarded after
        
        # Convert back to bool directly
        return warped_mask > threshold
    
    def forward(
        self,
        x,
        mask_features,
        mask=None,
        initial_query_feat: Optional[torch.Tensor] = None,
        initial_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with optional query propagation and alpha-blended attention masks.
        
        Args:
            x: Multi-scale features from pixel decoder
            mask_features: Mask features from pixel decoder
            mask: Optional attention mask (unused, kept for API compatibility)
            initial_query_feat: If provided, use these as initial query features
                               instead of learnable self.query_feat. Shape: [Q, B, D]
            initial_attn_mask: If provided, use for alpha-blended attention guidance.
                               This should be the reference view's predicted masks
                               WARPED to target view coordinates. Shape: [B, Q, H', W']
                               Values should be soft masks (0-1), where LOW values
                               indicate regions to ATTEND to.
        
        Returns:
            Dictionary with pred_logits, pred_masks, aux_outputs, and query_embeddings
        """
        dec = self.decoder
        
        assert len(x) == dec.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(dec.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(dec.pe_layer(x[i], None).flatten(2).permute(2, 0, 1))
            src.append((dec.input_proj[i](x[i]).flatten(2) + dec.level_embed.weight[i][None, :, None]).permute(2, 0, 1))

            # Flatten mask_features and move to src list if needed? 
            # Original code doesn't flatten mask_features here, it uses it in forward_prediction_heads

        _, bs, _ = src[0].shape
        num_heads = dec.num_heads if hasattr(dec, 'num_heads') else 8
        num_layers = dec.num_layers
        
        # Check if we're doing query propagation with warped masks
        use_alpha_blending = (initial_attn_mask is not None and initial_query_feat is not None)

        # QxNxC - positional embeddings for queries
        query_embed = dec.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
        # Use provided initial queries or default learnable ones
        if initial_query_feat is not None:
             # Add learnable query embedding to the propagated content query for positional info?
             # Standard Mask2Former: output = tgt + query_embed
             # Here initial_query_feat is 'output' from ref view.
             # We should probably reset it to just the content part?
             # For now, following standard: output IS the query feature.
             output = initial_query_feat.clone()
        else:
            output = dec.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # Prediction heads on initial query features (layer 0)
        outputs_class, outputs_mask, predicted_attn_mask = dec.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # Initial attention mask with alpha blending for first layer
        if use_alpha_blending:
            # Prepare warped mask for layer 0 resolution
            warped_attn_mask_l0 = self._prepare_warped_mask_for_layer(
                initial_attn_mask, 
                target_size=size_list[0], 
                num_heads=num_heads, 
                bs=bs
            )
            
            # Layer 0: Use PURE warped mask — no blending with predicted mask.
            # At layer 0, predicted_attn_mask comes from applying Q_ref (reference
            # queries) to target view features, producing geometrically misaligned
            # masks. Even with alpha=1.0, the blend function introduces floating-point
            # noise at threshold boundaries from the misaligned mask. Using the pure
            # warped geometric prior gives the cleanest spatial guidance for layer 0.
            attn_mask = warped_attn_mask_l0 > 0.5  # Direct bool conversion, no blend
            
            # Free memory
            del warped_attn_mask_l0
            
        else:
            attn_mask = predicted_attn_mask

        # Clean up early tensors to save memory
        # In loop, we accumulate predictions. If num_layers is large (9), this adds up.
        # But we need them for aux loss.

        for i in range(num_layers):
            level_index = i % dec.num_feature_levels
            
            if use_alpha_blending:
                decay_rate = 1.0 / num_layers
                alpha = max(0.0, 1.0 - (i * decay_rate))
            
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # Attention modification for masked attention 
            # (True = ignore, False = attend)
            # Mask2Former adds -inf to True positions.

            # Use gradient checkpointing for transformer layers to save memory
            # This reduces peak VRAM by ~30-40% at the cost of ~20% slower training
            if self.training and False:
                # Cross attention layer
                output = torch.utils.checkpoint.checkpoint(
                    dec.transformer_cross_attention_layers[i],
                    output, src[level_index],
                    attn_mask, None,
                    pos[level_index], query_embed,
                    use_reentrant=False
                )

                # Self attention layer
                output = torch.utils.checkpoint.checkpoint(
                    dec.transformer_self_attention_layers[i],
                    output, None,
                    None, 
                    query_embed,
                    use_reentrant=False
                )
                    
                # FFN Layer
                output = torch.utils.checkpoint.checkpoint(
                    dec.transformer_ffn_layers[i],
                    output,
                    use_reentrant=False
                )
            else:
                # During inference, no checkpointing needed
                output = dec.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    attn_mask, None,
                    pos[level_index], query_embed
                )
                output = dec.transformer_self_attention_layers[i](
                    output, None,
                    None, 
                    query_embed
                )
                output = dec.transformer_ffn_layers[i](output)

            # Prediction heads (no checkpointing - lightweight)
            outputs_class, outputs_mask, predicted_attn_mask = dec.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=list(size_list[(i + 1) % dec.num_feature_levels])
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # Prepare attention mask for next layer
            if use_alpha_blending and i < num_layers - 1:
                # Prepare warped mask for NEXT layer's resolution
                next_size = size_list[(i + 1) % dec.num_feature_levels]
                
                warped_attn_mask_next = self._prepare_warped_mask_for_layer(
                    initial_attn_mask, 
                    target_size=next_size, 
                    num_heads=num_heads, 
                    bs=bs
                )
                
                # Blend for next layer
                # Alpha decays: 1.0 -> ... -> 0.0
                next_alpha = max(0.0, 1.0 - ((i + 1) * decay_rate))
                
                attn_mask = self._blend_attention_masks(
                    warped_attn_mask_next,
                    predicted_attn_mask,
                    alpha=next_alpha
                )
                del warped_attn_mask_next
            else:
                attn_mask = predicted_attn_mask

        assert len(predictions_class) == num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': dec._set_aux_loss(
                predictions_class if dec.mask_classification else None, predictions_mask
            ),
            # Return refined query features for propagation to other views
            'query_embeddings': output,  # [Q, B, D]
        }
        return out


@META_ARCH_REGISTRY.register()
class MultiViewMask2Former(nn.Module):
    """
    Multi-View Mask2Former with Query Propagation.
    
    Key Architecture (following MapAnything paradigm):
    1. SINGLE shared Mask2Former head (pixel decoder + transformer decoder)
    2. Query propagation: Reference view queries reused across all views
    3. Same head applied to all views → unified semantic space
    
    Training Flow:
    1. Reference View (randomly selected):
       - Run full Mask2Former with learnable queries
       - Get refined queries Q_ref after transformer decoder
       - Compute standard Mask2Former loss
    
    2. Target Views (all other views):
       - Run SAME Mask2Former head with Q_ref as initial queries
       - Queries retain identity across views
       - Compute Mask2Former loss (same GT labels per query)
    
    This ensures Query #1 in View A is semantically identical to Query #1 in View B.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.num_views = cfg.MODEL.MULTIVIEW.NUM_VIEWS
        self.warmup_iter = cfg.MODEL.MULTIVIEW.WARMUP_ITER
        
        self.register_buffer('_iter', torch.tensor(0))
        
        # Multi-view backbone: processes all views, returns features for each
        self.backbone = MapAnythingMultiViewBackbone(cfg, None)
        
        # SINGLE shared Mask2Former head (pixel decoder + transformer decoder)
        # This head is applied to ALL views with query propagation
        backbone_shape = self.backbone.output_shape()
        self.sem_seg_head = MaskFormerHead(cfg, backbone_shape)
        
        # Wrap the transformer decoder for query propagation
        self.query_propagation_decoder = QueryPropagationTransformerDecoder(
            self.sem_seg_head.predictor
        )
        
        # Build the loss criterion (Hungarian matching + losses)
        self.criterion = self._build_criterion(cfg)
        
        # Store config for prepare_targets
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        
        # Store base weights for progressive dice loss warmup
        self._base_dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        self._base_mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        self._base_class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        
        # NO dice warmup - activate from iter 0, matching standard Mask2Former
        self._dice_warmup_start = 0     # Dice active from iter 0
        self._dice_warmup_end = 0       # No warmup period
        
        # =================================================================
        # MASK LOGIT BIAS INITIALIZATION
        # =================================================================
        # Problem: With default init, mask logits start near zero. BCE loss on
        # unmatched queries (80+ of 100) pushes ALL logits strongly negative
        # ("predict nothing" = low BCE). Dice loss can't recover because its
        # gradients vanish when predictions ≈ 0 (sigmoid(-4) ≈ 0.018).
        #
        # Fix: Initialize the last layer of mask_embed MLP so that initial
        # mask logits are slightly positive. The mask logit is computed as:
        #   logit = einsum("bqc,bchw->bqhw", mask_embed_output, mask_features)
        # With Xavier init on weights and bias=0, the expected logit is ~0.
        # We init bias to +1.0/sqrt(dim) ≈ +0.0625 per dimension, so the
        # aggregate dot product bias contribution is positive.
        # Also scale down weights to reduce logit variance and prevent extreme values.
        # =================================================================
        mask_embed = self.sem_seg_head.predictor.mask_embed
        last_layer = mask_embed.layers[-1]  # Last linear layer of MLP
        embed_dim = last_layer.bias.shape[0]  # 256
        with torch.no_grad():
            # Smaller weight scale → lower variance logits → more stable training
            nn.init.xavier_uniform_(last_layer.weight, gain=0.5)
            # Positive bias → mask_embed output has positive mean → positive logit mean
            nn.init.constant_(last_layer.bias, 1.0 / (embed_dim ** 0.5))
        print(f"  ✓ Mask embed last layer: Xavier(gain=0.5) + bias={1.0/(embed_dim**0.5):.4f} (anti-collapse)")
        
        # Log the learnable mask_logit_bias (defined in transformer decoder __init__)
        predictor = self.sem_seg_head.predictor
        if hasattr(predictor, 'mask_logit_bias'):
            bias_val = predictor.mask_logit_bias.data.mean().item()
            print(f"  ✓ Mask logit bias: {bias_val:.2f} (learnable, per-query, anti-collapse)")
        
        print(f"Multi-View Mask2Former initialized:")
        print(f"  - Num views: {self.num_views}")
        print(f"  - Num classes: {self.num_classes}")
        print(f"  - Single shared head: MaskFormerHead with Query Propagation")
        print(f"  - Warmup iterations: {self.warmup_iter}")
        print(f"  - Loss weights: CE={self._base_class_weight}, Mask={self._base_mask_weight}, Dice={self._base_dice_weight}")
        print(f"  - Dice active from iter 0 (no warmup)")
    
    @property
    def device(self):
        return next(self.parameters()).device

    def _build_criterion(self, cfg):
        """Build the Mask2Former loss criterion."""
        # Loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        
        # Build matcher
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        # Weight dict
        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }
        
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        # Build criterion
        criterion = SetCriterion(
            self.sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=["labels", "masks"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        
        return criterion
    
    def _get_dynamic_loss_weights(self) -> Dict[str, float]:
        """
        Get loss weights. All losses active from iter 0 (standard Mask2Former).
        
        Weights: CE=1.0, Mask=5.0, Dice=3.0
        - Dice is dominant (3.0) to force foreground prediction
        - CE at 1.0 matches Hungarian matcher cost_class=1.0
        - No warmup: matches standard Mask2Former and working single-view script
        """
        weights = {
            'loss_ce': self._base_class_weight,
            'loss_mask': self._base_mask_weight,
            'loss_dice': self._base_dice_weight,
        }
        
        return weights
    
    def _prepare_features_for_view(
        self,
        all_view_features: Dict[str, torch.Tensor],
        view_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for a specific view from all-view features.
        
        Args:
            all_view_features: Dict with {res2, res3, res4, res5} each [N, B, C, H', W']
            view_idx: Which view to extract
        
        Returns:
            Dict with {res2, res3, res4, res5} each [B, C, H', W']
        """
        view_features = {}
        for key in ['res2', 'res3', 'res4', 'res5']:
            view_features[key] = all_view_features[key][view_idx]  # [B, C, H', W']
        return view_features
    
    def forward(self, batched_inputs: Dict) -> Dict:
        """
        Forward pass with query propagation across views.
        
        Training:
        1. Select random reference view
        2. Run Mask2Former on reference view → get refined queries
        3. Run SAME Mask2Former on target views with propagated queries
        4. Compute loss on all views
        
        Inference:
        Run on reference view (view 0) only.
        
        Args:
            batched_inputs: Output from multi_view_collate_fn
        
        Returns:
            Dictionary of losses (training) or predictions (inference)
        """
        # Move inputs to device (collate_fn produces CPU tensors)
        images = batched_inputs['images'].to(self.device)
        batched_inputs['images'] = images
        
        if 'camera_poses' in batched_inputs:
            batched_inputs['camera_poses'] = batched_inputs['camera_poses'].to(self.device)
            
        if 'camera_intrinsics' in batched_inputs:
            batched_inputs['camera_intrinsics'] = batched_inputs['camera_intrinsics'].to(self.device)
            
        if 'depth' in batched_inputs and batched_inputs['depth'] is not None:
             batched_inputs['depth'] = batched_inputs['depth'].to(self.device)

        # Move GT instances to device
        if 'instances' in batched_inputs:
            batched_inputs['instances'] = [x.to(self.device) for x in batched_inputs['instances']]

        camera_poses = batched_inputs.get('camera_poses')
        camera_intrinsics = batched_inputs.get('camera_intrinsics')
        
        B, N, C, H, W = images.shape
        
        # Run multi-view backbone → get features for ALL views
        backbone_out = self.backbone(
            images=images,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            return_all_views=True,
        )
        
        all_view_features = backbone_out['all_view_features']  # {res2: [N,B,C,H,W], ...}
        
        if self.training:
            return self._forward_train(
                all_view_features=all_view_features,
                batched_inputs=batched_inputs,
                B=B, N=N, H=H, W=W,
            )
        else:
            return self._forward_inference(
                all_view_features=all_view_features,
                B=B, N=N,
            )
    
    def _run_pixel_decoder(self, features: Dict[str, torch.Tensor]):
        """Run pixel decoder on features, return mask_features and multi_scale_features."""
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.sem_seg_head.pixel_decoder.forward_features(features)
        return mask_features, multi_scale_features
    
    def prepare_targets(self, targets: List, h_pad: int, w_pad: int) -> List[Dict]:
        """
        Prepare targets for criterion (convert Instances to dict format).
        
        Args:
            targets: List of Instances objects with gt_classes and gt_masks
            h_pad: Padded height
            w_pad: Padded width
        
        Returns:
            List of dicts with 'labels' and 'masks' keys
        """
        new_targets = []
        for targets_per_image in targets:
            # Get GT masks and classes
            gt_masks = targets_per_image.gt_masks
            
            # Pad GT masks to match image size
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device
            )
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
        return new_targets
    

    def _upsample_predictions(self, outputs, target_size):
        """Deprecated: Do not use. Upsampling dense masks causes OOM."""
        pass

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Mask2Former losses using SetCriterion.
        
        Uses dynamic loss weights for progressive dice loss warmup.
        
        Args:
            outputs: Dict containing 'pred_logits' and 'pred_masks' from decoder
            targets: List of Instances objects (one per batch element)
            target_size: (H, W) tuple specifying the spatial resolution for targets.
                         If None, uses the prediction mask size (which might be small).
        
        Returns:
            Dict of losses with weight_dict applied
        """
        # Get output mask shape for padding targets
        if target_size is not None:
            h_pad, w_pad = target_size
        else:
            pred_masks = outputs["pred_masks"]  # [B, Q, H', W']
            h_pad, w_pad = pred_masks.shape[-2:]
        
        # Prepare targets in criterion format
        prepared_targets = self.prepare_targets(targets, h_pad, w_pad)
        
        # Clamp mask logits before loss to prevent fp16 overflow
        if 'pred_masks' in outputs:
            outputs['pred_masks'] = outputs['pred_masks'].clamp(min=-20.0, max=20.0)
        if 'aux_outputs' in outputs:
            for aux in outputs['aux_outputs']:
                if 'pred_masks' in aux:
                    aux['pred_masks'] = aux['pred_masks'].clamp(min=-20.0, max=20.0)
        
        # Compute raw losses using criterion
        losses = self.criterion(outputs, prepared_targets)
        
        # NaN safety: replace any NaN losses with zero to prevent crash
        for k in list(losses.keys()):
            if torch.isnan(losses[k]).any() or torch.isinf(losses[k]).any():
                print(f"  ⚠️  NaN/Inf detected in {k}, zeroing out")
                losses[k] = torch.zeros_like(losses[k])
        
        # Get dynamic weights (progressive dice loss)
        dynamic_weights = self._get_dynamic_loss_weights()
        
        # Apply dynamic weight dict instead of static weight_dict
        for k in list(losses.keys()):
            # Map auxiliary loss keys to base loss keys for weight lookup
            base_key = k.split('_')[0] + '_' + k.split('_')[1] if '_' in k else k
            # Handle aux losses like "loss_ce_0", "loss_mask_1", etc.
            if base_key.startswith('loss_') and len(k.split('_')) > 2:
                base_key = '_'.join(k.split('_')[:2])  # e.g., "loss_ce"
            
            if base_key in dynamic_weights:
                losses[k] *= dynamic_weights[base_key]
            elif k in self.criterion.weight_dict:
                # Fallback to static weight for unrecognized losses
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # Remove losses not in weight dict
                losses.pop(k)
        
        return losses
    
    def _forward_train(
        self,
        all_view_features: Dict[str, torch.Tensor],
        batched_inputs: Dict,
        B: int, N: int, H: int, W: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with query propagation and spatial bridging.
        
        Key steps:
        1. Randomly select reference view
        2. Run Mask2Former on reference → get refined queries + predicted masks
        3. SPATIAL BRIDGING: Warp reference masks to target view using depth + poses
        4. Propagate queries + warped attention masks to target views
        5. Aggregate losses from all views
        """
        # Randomly select reference view (different each iteration for robustness)
        ref_view_idx = torch.randint(0, N, (1,)).item()
        
        # Get camera poses and intrinsics for geometric warping
        camera_poses = batched_inputs['camera_poses']  # [B, N, 4, 4]
        camera_intrinsics = batched_inputs['camera_intrinsics']  # [B, N, 3, 3]
        
        # Get depth if available (for spatial bridging)
        # depth: [B, N, 1, H, W] or None
        depth = batched_inputs.get('depth', None)
        has_depth = depth is not None
        
        # Get ground truth instances for all views
        # batched_inputs['instances'] is [B * N] list
        all_instances = batched_inputs['instances']  # List of length B * N
        
        # Reorganize instances: instances_per_view[view_idx] = list of B instances
        # Collate order is [b0v0, b0v1, ..., b0v(N-1), b1v0, b1v1, ...] (batch-major)
        # So index for batch b, view v is: b * N + v
        instances_per_view = []
        for v_idx in range(N):
            view_instances = []
            for b_idx in range(B):
                idx = b_idx * N + v_idx
                view_instances.append(all_instances[idx])
            instances_per_view.append(view_instances)
        
        # ========================================
        # Step 1: Reference View - Full Mask2Former
        # ========================================
        ref_features = self._prepare_features_for_view(all_view_features, ref_view_idx)
        ref_gt = instances_per_view[ref_view_idx]
        
        # Run pixel decoder on reference view
        ref_mask_features, ref_multi_scale_features = self._run_pixel_decoder(ref_features)
        
        # Run transformer decoder with learnable queries (no propagation for reference)
        ref_outputs = self.query_propagation_decoder(
            ref_multi_scale_features,
            ref_mask_features,
            mask=None,
            initial_query_feat=None,  # Use learnable queries for reference view
            initial_attn_mask=None,   # No warped mask for reference
        )

        # Get refined queries for propagation
        ref_queries = ref_outputs['query_embeddings']  # [Q, B, D]
        ref_pred_masks = ref_outputs['pred_masks']     # [B, Q, H, W]
        
        # Get reference view's pose and intrinsics
        ref_pose = camera_poses[:, ref_view_idx]      # [B, 4, 4]
        ref_intrinsic = camera_intrinsics[:, ref_view_idx]  # [B, 3, 3]
        ref_depth = depth[:, ref_view_idx] if has_depth else None  # [B, 1, H, W]
        
        # Compute loss for reference view
        # We pass target_size=(H, W) to ensure targets are prepared at full resolution
        # for point supervision, while predictions remain at low resolution (stride 4).
        ref_losses = self._compute_losses(ref_outputs, ref_gt, target_size=(H, W))
        
        # ========================================
        # Step 2: Target Views - Query Propagation with Spatial Bridging
        # ========================================
        target_losses_list = []
        target_outputs_list = []
        target_gt_list = []
        
        # Only do query propagation after warmup
        do_propagation = (self._iter >= self.warmup_iter)
        
        for v_idx in range(N):
            if v_idx == ref_view_idx:
                continue  # Skip reference view
            
            target_features = self._prepare_features_for_view(all_view_features, v_idx)
            target_gt = instances_per_view[v_idx]
            target_gt_list.append(target_gt)
            
            # Run pixel decoder on target view
            target_mask_features, target_multi_scale_features = self._run_pixel_decoder(target_features)
            
            if do_propagation:
                # Get target view's pose
                tgt_pose = camera_poses[:, v_idx]      # [B, 4, 4]
                tgt_depth = depth[:, v_idx] if has_depth else None # [B, 1, H, W]
                
                # SPATIAL BRIDGING: Warp reference masks to target view
                warped_attn_mask = None
                if has_depth and ref_depth is not None:
                    # Create warped attention mask for target decoder
                    target_H, target_W = target_multi_scale_features[0].shape[-2:]
                    warped_attn_mask = create_warped_attention_mask(
                        ref_pred_masks=ref_pred_masks,
                        ref_depth=ref_depth,
                        ref_pose=ref_pose,
                        tgt_pose=tgt_pose,
                        intrinsics=ref_intrinsic,  # Assume same intrinsics
                        target_size=(target_H, target_W),
                        tgt_depth=tgt_depth,       # Pass target depth for robust backward warping
                        mask_threshold=0.5,
                    )
                
                # KEY: Propagate queries + warped attention mask from reference view
                target_outputs = self.query_propagation_decoder(
                    target_multi_scale_features,
                    target_mask_features,
                    mask=None,
                    initial_query_feat=ref_queries,      # Use reference view's refined queries
                    initial_attn_mask=warped_attn_mask,  # Use warped mask for spatial bridging
                )
            else:
                # During warmup, use learnable queries (each view learns independently)
                target_outputs = self.query_propagation_decoder(
                    target_multi_scale_features,
                    target_mask_features,
                    mask=None,
                    initial_query_feat=None,
                    initial_attn_mask=None,
                )
            
            target_loss = self._compute_losses(target_outputs, target_gt, target_size=(H, W))
            target_losses_list.append(target_loss)
            target_outputs_list.append(target_outputs)
        
        # ========================================
        # Step 3: Aggregate Losses
        # ========================================
        total_losses = {}
        
        # Reference view losses (full weight)
        for key, val in ref_losses.items():
            total_losses[key] = val
        
        # Target view losses (averaged)
        if target_losses_list:
            for key in target_losses_list[0].keys():
                target_loss_sum = sum(tl[key] for tl in target_losses_list)
                target_loss_avg = target_loss_sum / len(target_losses_list)
                # Add as separate loss for monitoring (DETACHED to avoid double counting in backward)
                total_losses[f'{key}_target'] = target_loss_avg.detach()
                # Also add to main loss (Backprop flows through here)
                total_losses[key] = total_losses.get(key, 0) + target_loss_avg
        
        # Store detailed metrics for analysis (optional, for debugging)
        if hasattr(self, '_store_view_metrics') and self._store_view_metrics:
            # All these must be DETACHED to avoid affecting total_loss or backward
            total_losses['_ref_view_idx'] = torch.tensor(ref_view_idx, dtype=torch.float32, device=self.device).detach()
            
            # Helper to safely get detached scalar
            def get_detached(d, k):
                val = d.get(k, torch.tensor(0.0, device=self.device))
                return val.detach()
            
            total_losses['_ref_loss_ce'] = get_detached(ref_losses, 'loss_ce')
            total_losses['_ref_loss_mask'] = get_detached(ref_losses, 'loss_mask')
            total_losses['_ref_loss_dice'] = get_detached(ref_losses, 'loss_dice')
            
            # Average target view losses
            if target_losses_list:
                total_losses['_target_loss_ce_avg'] = sum(get_detached(tl, 'loss_ce') for tl in target_losses_list) / len(target_losses_list)
                total_losses['_target_loss_mask_avg'] = sum(get_detached(tl, 'loss_mask') for tl in target_losses_list) / len(target_losses_list)
                total_losses['_target_loss_dice_avg'] = sum(get_detached(tl, 'loss_dice') for tl in target_losses_list) / len(target_losses_list)
        
        # ========================================
        # Gradient Diagnostics (every 20 iters)
        # ========================================
        current_iter = self._iter.item() if isinstance(self._iter, torch.Tensor) else self._iter
        if current_iter % 20 == 0 and current_iter > 0:
            # Get dynamic weights for logging
            dynamic_weights = self._get_dynamic_loss_weights()
            
            # Store loss diagnostics for logging hook
            loss_ce_val = total_losses.get('loss_ce', torch.tensor(0.0))
            loss_mask_val = total_losses.get('loss_mask', torch.tensor(0.0)) 
            loss_dice_val = total_losses.get('loss_dice', torch.tensor(0.0))
            
            print(f"\n=== LOSS DIAGNOSTICS (iter {current_iter}) ===")
            print(f"Dynamic Weights: CE={dynamic_weights['loss_ce']:.2f}, Mask={dynamic_weights['loss_mask']:.2f}, Dice={dynamic_weights['loss_dice']:.3f}")
            print(f"Weighted Losses: CE={loss_ce_val.item():.4f}, Mask={loss_mask_val.item():.4f}, Dice={loss_dice_val.item():.4f}")
            
            # Check mask logits statistics from reference output
            if 'pred_masks' in ref_outputs:
                mask_logits = ref_outputs['pred_masks']
                print(f"Mask Logits: min={mask_logits.min().item():.2f}, max={mask_logits.max().item():.2f}, mean={mask_logits.mean().item():.2f}")
                # Check if masks are collapsing (all very negative = always 0 after sigmoid)
                if mask_logits.max().item() < -10:
                    print("  ⚠️  WARNING: Mask logits very negative - masks may be collapsing!")
        
        # Final NaN/Inf safety: zero out any NaN losses to prevent crash
        nan_detected = False
        for k in list(total_losses.keys()):
            if k.startswith('_'):
                continue  # Skip diagnostic keys
            val = total_losses[k]
            if isinstance(val, torch.Tensor) and (torch.isnan(val).any() or torch.isinf(val).any()):
                print(f"  ⚠️  NaN/Inf in total_losses['{k}'], zeroing out")
                total_losses[k] = torch.zeros_like(val)
                nan_detected = True
        if nan_detected:
            print(f"  ⚠️  NaN detected at iter {current_iter if current_iter % 20 == 0 else self._iter.item()}, skipping this batch gracefully")
        
        # Update iteration counter
        self._iter += 1
        
        return total_losses
    
    def compute_view_metrics(
        self,
        batched_inputs: Dict,
        compute_class_probs: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute detailed per-view metrics for evaluation/analysis.
        
        This computes:
        1. Per-view losses (ref vs targets)
        2. Class probability consistency across views
        3. Query-level prediction similarity
        
        Args:
            batched_inputs: Same format as forward()
            compute_class_probs: Whether to compute class probability analysis
        
        Returns:
            Dict with detailed metrics:
            - ref_view_losses: Dict of losses for reference view
            - target_view_losses: List of dicts, one per target view            - class_prob_consistency: KL divergence between views (if enabled)
            - query_prediction_similarity: Cosine similarity of query features
        """
        images = batched_inputs['images']  # [B, N, 3, H, W]
        camera_poses = batched_inputs['camera_poses']
        camera_intrinsics = batched_inputs['camera_intrinsics']
        depth = batched_inputs.get('depth', None)
        all_instances = batched_inputs['instances']
        
        B, N, C, H, W = images.shape
        
        # Run backbone
        backbone_out = self.backbone(
            images=images,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            return_all_views=True,
        )
        all_view_features = backbone_out['all_view_features']
        
        # Reorganize instances
        # Collate order is [b0v0, b0v1, ..., b0v(N-1), b1v0, b1v1, ...] (batch-major)
        instances_per_view = []
        for v_idx in range(N):
            view_instances = []
            for b_idx in range(B):
                idx = b_idx * N + v_idx
                view_instances.append(all_instances[idx])
            instances_per_view.append(view_instances)
        
        # Use view 0 as reference for evaluation
        ref_view_idx = 0
        ref_features = self._prepare_features_for_view(all_view_features, ref_view_idx)
        ref_gt = instances_per_view[ref_view_idx]
        
        # Run reference view
        ref_mask_features, ref_multi_scale_features = self._run_pixel_decoder(ref_features)
        ref_outputs = self.query_propagation_decoder(
            ref_multi_scale_features,
            ref_mask_features,
            mask=None,
            initial_query_feat=None,
            initial_attn_mask=None,
        )
        
        ref_queries = ref_outputs['query_embeddings']  # [Q, B, D]
        ref_pred_masks = ref_outputs['pred_masks']
        ref_pred_logits = ref_outputs['pred_logits']  # [B, Q, C+1]
        
        # Compute reference view losses
        ref_losses = self._compute_losses(ref_outputs, ref_gt)
        
        # Get reference view pose/depth for warping
        ref_pose = camera_poses[:, ref_view_idx]
        ref_intrinsic = camera_intrinsics[:, ref_view_idx]
        ref_depth = depth[:, ref_view_idx] if depth is not None else None
        has_depth = ref_depth is not None
        
        # Process target views
        target_view_losses = []
        target_view_logits = []
        
        for v_idx in range(N):
            if v_idx == ref_view_idx:
                continue
            
            target_features = self._prepare_features_for_view(all_view_features, v_idx)
            target_gt = instances_per_view[v_idx]
            
            # Run pixel decoder
            target_mask_features, target_multi_scale_features = self._run_pixel_decoder(target_features)
            
            # Warp attention mask if depth available
            warped_attn_mask = None
            if has_depth and ref_depth is not None:
                tgt_pose = camera_poses[:, v_idx]
                target_H, target_W = target_multi_scale_features[0].shape[-2:]
                warped_attn_mask = create_warped_attention_mask(
                    ref_pred_masks=ref_pred_masks,
                    ref_depth=ref_depth,
                    ref_pose=ref_pose,
                    tgt_pose=tgt_pose,
                    intrinsics=ref_intrinsic,
                    target_size=(target_H, target_W),
                    mask_threshold=0.5,
                )
            
            # Run with propagated queries
            target_outputs = self.query_propagation_decoder(
                target_multi_scale_features,
                target_mask_features,
                mask=None,
                initial_query_feat=ref_queries,
                initial_attn_mask=warped_attn_mask,
            )
            
            target_pred_logits = target_outputs['pred_logits']  # [B, Q, C+1]
            
            # Compute losses
            target_loss = self._compute_losses(target_outputs, target_gt)
            target_view_losses.append({
                'view_idx': v_idx,
                'loss_ce': target_loss.get('loss_ce', torch.tensor(0.0)).item(),
                'loss_mask': target_loss.get('loss_mask', torch.tensor(0.0)).item(),
                'loss_dice': target_loss.get('loss_dice', torch.tensor(0.0)).item(),
                'total': sum(v for k, v in target_loss.items() if not k.startswith('loss_')).item(),
            })
            
            target_view_logits.append(target_pred_logits)
        
        # Compute class probability consistency
        metrics = {
            'ref_view_losses': {
                'loss_ce': ref_losses.get('loss_ce', torch.tensor(0.0)).item(),
                'loss_mask': ref_losses.get('loss_mask', torch.tensor(0.0)).item(),
                'loss_dice': ref_losses.get('loss_dice', torch.tensor(0.0)).item(),
            },
            'target_view_losses': target_view_losses,
        }
        
        if compute_class_probs and target_view_logits:
            # Compute class probability consistency using KL divergence
            ref_probs = F.softmax(ref_pred_logits, dim=-1)  # [B, Q, C+1]
            
            kl_divs = []
            for tgt_logits in target_view_logits:
                tgt_probs = F.softmax(tgt_logits, dim=-1)  # [B, Q, C+1]
                
                # KL divergence per query, averaged over batch
                kl_div = F.kl_div(
                    tgt_probs.log(),
                    ref_probs,
                    reduction='batchmean'
                )
                kl_divs.append(kl_div.item())
            
            metrics['class_prob_kl_divergence'] = {
                'mean': np.mean(kl_divs),
                'std': np.std(kl_divs),
                'per_view': kl_divs,
            }
            
            # Compute top-k class agreement
            ref_top_classes = ref_pred_logits.argmax(dim=-1)  # [B, Q]
            class_agreements = []
            for tgt_logits in target_view_logits:
                tgt_top_classes = tgt_logits.argmax(dim=-1)  # [B, Q]
                agreement = (ref_top_classes == tgt_top_classes).float().mean().item()
                class_agreements.append(agreement)
            
            metrics['top_class_agreement'] = {
                'mean': np.mean(class_agreements),
                'std': np.std(class_agreements),
                'per_view': class_agreements,
            }
        
        return metrics
    
    def _forward_inference(
        self,
        all_view_features: Dict[str, torch.Tensor],
        B: int, N: int,
    ) -> Dict:
        """
        Inference forward pass.
        
        Use view 0 as reference. Can optionally propagate to other views
        for multi-view consistent predictions.
        """
        # Use view 0 as reference for inference
        ref_features = self._prepare_features_for_view(all_view_features, 0)
        
        # Run Mask2Former head
        results = self.sem_seg_head(ref_features, None)
        
        return results


# ============================================================
# CONFIGURATION
# ============================================================

def add_multiview_config(cfg):
    """Add multi-view specific configuration options."""
    cfg.MODEL.MULTIVIEW = CN()
    cfg.MODEL.MULTIVIEW.NUM_VIEWS = 2  # Number of views per scene
    cfg.MODEL.MULTIVIEW.USE_POSES = True  # Whether to use camera poses
    
    # Query propagation settings
    cfg.MODEL.MULTIVIEW.WARMUP_ITER = 1000  # Iterations before enabling query propagation
    
    # View selection parameters
    cfg.MODEL.MULTIVIEW.MIN_CAMERA_DISTANCE = 0.3  # Minimum distance between cameras (meters)
    cfg.MODEL.MULTIVIEW.MAX_CAMERA_DISTANCE = 5.0  # Maximum distance between cameras
    cfg.MODEL.MULTIVIEW.MIN_OVERLAP = 0.2  # Minimum required overlap between views
    cfg.MODEL.MULTIVIEW.MAX_OVERLAP = 0.8  # Maximum allowed overlap
    
    # MapAnything checkpoint path
    cfg.MODEL.MAPANYTHING = CN()
    cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH = ""  # Path to pretrained MapAnything checkpoint
    
    # Depth loading options (for spatial bridging with warped attention masks)
    cfg.INPUT.LOAD_DEPTH = True  # Load/render depth from mesh for alpha blending
    cfg.INPUT.DEPTH_CACHE_DIR = None  # Optional global cache dir for rendered depths
    
    # Dataset paths
    cfg.DATASETS.SCANNETPP_ROOT = ""  # Root directory of ScanNet++ dataset
    cfg.DATASETS.SCANNETPP_PANOPTIC_ROOT = ""  # Directory with panoptic annotations


def setup_cfg(args):
    """Setup configuration for multi-view training."""
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_multiview_config(cfg)
    
    # Add DPT LR config (Must be defined BEFORE merging from file)
    cfg.SOLVER.DPT_LR = 1e-5
    
    # Merge from file if provided
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Enable AMP like in single-view training (Force it AFTER merge to avoid overwrite)
    cfg.SOLVER.AMP.ENABLED = True

    # Set dataset paths from args if provided
    try:
        cfg.defrost()
    except:
        pass

    if hasattr(args, 'scannetpp_root') and args.scannetpp_root:
        cfg.DATASETS.SCANNETPP_ROOT = args.scannetpp_root
        if hasattr(args, 'panoptic_root') and args.panoptic_root:
            cfg.DATASETS.SCANNETPP_PANOPTIC_ROOT = args.panoptic_root
        else:
            cfg.DATASETS.SCANNETPP_PANOPTIC_ROOT = os.path.join(
                os.path.dirname(args.scannetpp_root), "panoptic"
            )
    
    # Enable Gradient Clipping to prevent NaN/inf gradients
    # IMPORTANT: Per-parameter norm clipping at 1.0 was too aggressive — it
    # saturated every step (all grad norms = exactly 1.0), killing the weak
    # dice gradients that need to resist the BCE "predict nothing" collapse.
    # Increase to 5.0 so clipping only fires during genuine gradient explosion,
    # not on every normal training step.
    try:
        cfg.defrost()
    except:
        pass
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0   # Relaxed: was 1.0 (saturating every step)
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Set multi-view backbone
    try:
        cfg.defrost()
    except:
        pass
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_multiview_backbone"
    
    cfg.freeze()
    return cfg


# ============================================================
# REGISTER BACKBONE
# ============================================================

from detectron2.modeling.backbone import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
def build_mapanything_multiview_backbone(cfg, input_shape):
    return MapAnythingMultiViewBackbone(cfg, input_shape)


# ============================================================
# CSV METRICS LOGGING HOOK
# ============================================================

class CSVMetricsLogger(HookBase):
    """
    Hook to log training losses and evaluation metrics to CSV and TensorBoard.
    
    Logs:
    - Training: total_loss, loss_ce, loss_mask, loss_dice (at configurable interval)
    - Evaluation: PQ, RQ, SQ (after each evaluation period)
    - View Analysis: Per-view losses and class probability consistency
    """
    
    def __init__(self, output_dir: str, log_period: int = 20):
        """
        Args:
            output_dir: Directory to save CSV files and TensorBoard logs
            log_period: How often to log training losses (in iterations)
        """
        self.output_dir = output_dir
        self.log_period = log_period
        
        # CSV file paths
        self.train_csv_path = os.path.join(output_dir, "training_metrics.csv")
        self.eval_csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        self.view_csv_path = os.path.join(output_dir, "view_analysis_metrics.csv")
        
        # TensorBoard writer
        tensorboard_dir = os.path.join(output_dir, "tensorboard")
        if comm.is_main_process():
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.tb_writer = None
        
        # Initialize CSV files
        self._train_csv_initialized = False
        self._eval_csv_initialized = False
        self._view_csv_initialized = False
    
    def _init_train_csv(self, loss_keys: List[str]):
        """Initialize training CSV with headers."""
        if comm.is_main_process() and not self._train_csv_initialized:
            with open(self.train_csv_path, 'w') as f:
                headers = ['iteration', 'total_loss'] + sorted(loss_keys) + ['lr']
                f.write(','.join(headers) + '\n')
            self._train_csv_initialized = True
    
    def _init_eval_csv(self):
        """Initialize evaluation CSV with headers."""
        if comm.is_main_process() and not self._eval_csv_initialized:
            with open(self.eval_csv_path, 'w') as f:
                headers = ['iteration', 'PQ', 'SQ', 'RQ', 'PQ_th', 'SQ_th', 'RQ_th', 'PQ_st', 'SQ_st', 'RQ_st']
                f.write(','.join(headers) + '\n')
            self._eval_csv_initialized = True
    
    def _init_view_csv(self):
        """Initialize view analysis CSV with headers."""
        if comm.is_main_process() and not self._view_csv_initialized:
            with open(self.view_csv_path, 'w') as f:
                headers = [
                    'iteration', 'scene_id', 'view_type', 'view_idx',
                    'loss_ce', 'loss_mask', 'loss_dice', 'total_loss',
                    'class_prob_kl', 'top_class_agreement'
                ]
                f.write(','.join(headers) + '\n')
            self._view_csv_initialized = True
    
    def after_step(self):
        """Log training losses periodically."""
        # Only log on main process and at specified intervals
        if not comm.is_main_process():
            return
        
        iteration = self.trainer.iter
        if iteration % self.log_period != 0:
            return
        
        # Get losses from storage
        storage = self.trainer.storage
        
        # Collect loss values
        loss_dict = {}
        total_loss = 0.0
        
        try:
            # Get all logged losses
            for key in storage.latest().keys():
                if key.startswith('loss'):
                    val = storage.latest()[key][0]  # (value, iteration) tuple
                    loss_dict[key] = val
                    if key == 'total_loss' or (not key.endswith('_target')):
                        total_loss += val
        except Exception:
            return  # Skip if storage not ready
        
        if not loss_dict:
            return
        
        # Initialize CSV if needed
        loss_keys = [k for k in loss_dict.keys() if k != 'total_loss']
        if not self._train_csv_initialized:
            self._init_train_csv(loss_keys)
        
        # Get learning rate
        lr = self.trainer.optimizer.param_groups[0]['lr']
        
        # Write to CSV
        try:
            with open(self.train_csv_path, 'a') as f:
                values = [str(iteration), f'{total_loss:.6f}']
                for key in sorted(loss_keys):
                    values.append(f'{loss_dict.get(key, 0.0):.6f}')
                values.append(f'{lr:.8f}')
                f.write(','.join(values) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write training metrics to CSV: {e}")
        
        # Write to TensorBoard
        if self.tb_writer is not None:
            try:
                # Log all losses
                self.tb_writer.add_scalar('train/total_loss', total_loss, iteration)
                for key in loss_keys:
                    self.tb_writer.add_scalar(f'train/{key}', loss_dict.get(key, 0.0), iteration)
                # Log learning rate
                self.tb_writer.add_scalar('train/learning_rate', lr, iteration)
                # Log GPU memory if available
                if 'max_mem' in storage.latest():
                    max_mem = storage.latest()['max_mem'][0]
                    self.tb_writer.add_scalar('system/max_gpu_memory_mb', max_mem, iteration)
            except Exception as e:
                print(f"Warning: Failed to write training metrics to TensorBoard: {e}")
    
    def after_eval(self, eval_results: Dict):
        """Log evaluation metrics after evaluation."""
        if not comm.is_main_process():
            return
        
        if not self._eval_csv_initialized:
            self._init_eval_csv()
        
        iteration = self.trainer.iter
        
        # Extract panoptic quality metrics
        # COCOPanopticEvaluator returns metrics under 'panoptic_seg' key
        try:
            metrics = eval_results.get('panoptic_seg', {})
            
            pq = metrics.get('PQ', 0.0)
            sq = metrics.get('SQ', 0.0)
            rq = metrics.get('RQ', 0.0)
            pq_th = metrics.get('PQ_th', 0.0)  # Things
            sq_th = metrics.get('SQ_th', 0.0)
            rq_th = metrics.get('RQ_th', 0.0)
            pq_st = metrics.get('PQ_st', 0.0)  # Stuff
            sq_st = metrics.get('SQ_st', 0.0)
            rq_st = metrics.get('RQ_st', 0.0)
            
            with open(self.eval_csv_path, 'a') as f:
                values = [
                    str(iteration),
                    f'{pq:.4f}', f'{sq:.4f}', f'{rq:.4f}',
                    f'{pq_th:.4f}', f'{sq_th:.4f}', f'{rq_th:.4f}',
                    f'{pq_st:.4f}', f'{sq_st:.4f}', f'{rq_st:.4f}',
                ]
                f.write(','.join(values) + '\n')
            
            print(f"\n{'='*60}")
            print(f"Evaluation at iteration {iteration}:")
            print(f"  PQ={pq:.2f}  SQ={sq:.2f}  RQ={rq:.2f}")
            print(f"  PQ_th={pq_th:.2f}  PQ_st={pq_st:.2f}")
            print(f"{'='*60}\n")
            
            # Write to TensorBoard
            if self.tb_writer is not None:
                try:
                    self.tb_writer.add_scalar('eval/PQ', pq, iteration)
                    self.tb_writer.add_scalar('eval/SQ', sq, iteration)
                    self.tb_writer.add_scalar('eval/RQ', rq, iteration)
                    self.tb_writer.add_scalar('eval/PQ_things', pq_th, iteration)
                    self.tb_writer.add_scalar('eval/SQ_things', sq_th, iteration)
                    self.tb_writer.add_scalar('eval/RQ_things', rq_th, iteration)
                    self.tb_writer.add_scalar('eval/PQ_stuff', pq_st, iteration)
                    self.tb_writer.add_scalar('eval/SQ_stuff', sq_st, iteration)
                    self.tb_writer.add_scalar('eval/RQ_stuff', rq_st, iteration)
                except Exception as e:
                    print(f"Warning: Failed to write evaluation metrics to TensorBoard: {e}")
            
        except Exception as e:
            print(f"Warning: Failed to log evaluation metrics: {e}")
    
    def log_view_analysis(
        self,
        iteration: int,
        scene_id: str,
        view_metrics: Dict[str, Any],
    ):
        """
        Log detailed per-view analysis metrics.
        
        Args:
            iteration: Current training iteration
            scene_id: Scene identifier
            view_metrics: Dict returned by compute_view_metrics()
        """
        if not comm.is_main_process():
            return
        
        if not self._view_csv_initialized:
            self._init_view_csv()
        
        try:
            with open(self.view_csv_path, 'a') as f:
                # Log reference view
                ref_losses = view_metrics['ref_view_losses']
                values = [
                    str(iteration),
                    scene_id,
                    'ref',
                    '0',
                    f"{ref_losses['loss_ce']:.6f}",
                    f"{ref_losses['loss_mask']:.6f}",
                    f"{ref_losses['loss_dice']:.6f}",
                    f"{sum(ref_losses.values()):.6f}",
                    '0.0',  # No KL for ref
                    '1.0',  # Perfect agreement with itself
                ]
                f.write(','.join(values) + '\n')
                
                # Log target views
                target_losses = view_metrics['target_view_losses']
                kl_divs = view_metrics.get('class_prob_kl_divergence', {}).get('per_view', [0.0] * len(target_losses))
                agreements = view_metrics.get('top_class_agreement', {}).get('per_view', [0.0] * len(target_losses))
                
                for i, tgt_loss in enumerate(target_losses):
                    values = [
                        str(iteration),
                        scene_id,
                        'target',
                        str(tgt_loss['view_idx']),
                        f"{tgt_loss['loss_ce']:.6f}",
                        f"{tgt_loss['loss_mask']:.6f}",
                        f"{tgt_loss['loss_dice']:.6f}",
                        f"{tgt_loss['total']:.6f}",
                        f"{kl_divs[i] if i < len(kl_divs) else 0.0:.6f}",
                        f"{agreements[i] if i < len(agreements) else 0.0:.6f}",
                    ]
                    f.write(','.join(values) + '\n')
                    
        except Exception as e:
            print(f"Warning: Failed to log view analysis metrics: {e}")
    
    def after_train(self):
        """Close TensorBoard writer after training completes."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
                logger.info("TensorBoard writer closed successfully")
            except Exception as e:
                print(f"Warning: Failed to close TensorBoard writer: {e}")


# ============================================================
# DATALOADER UTILITIES (module-level for pickling with spawn)
# ============================================================

class FilteredDataset:
    """
    Dataset wrapper that filters out None samples by retrying.
    
    This class is defined at module level to be picklable for
    multiprocessing with spawn start method.
    """
    def __init__(self, base_dataset):
        self._base = base_dataset
        
    def __len__(self):
        return len(self._base)
    
    def __getitem__(self, idx):
        """Get item, retrying with next index if None is returned."""
        max_retries = 50  # Avoid infinite loop
        attempts = 0
        
        while attempts < max_retries:
            try:
                item = self._base[idx]
                if item is not None:
                    return item
                # If None, try next index (wraparound)
                idx = (idx + 1) % len(self._base)
                attempts += 1
            except Exception as e:
                # If error, also try next index
                logger.warning(f"Error loading sample {idx}: {e}")
                idx = (idx + 1) % len(self._base)
                attempts += 1
        
        # If all retries failed, raise error
        raise RuntimeError(
            f"Failed to load valid sample after {max_retries} attempts. "
            f"Check if dataset has valid samples with camera data."
        )
    
    def __iter__(self):
        for item in self._base:
            if item is not None:
                yield item
            # Skip None samples silently in iteration mode


def dataloader_worker_init_fn(worker_id):
    """
    Initialize worker with proper seed and disable problematic libraries.
    
    This function is defined at module level to be picklable for
    multiprocessing with spawn start method.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    # Disable OpenCV threading in workers
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


# ============================================================
# GRADIENT DIAGNOSTICS HOOK
# ============================================================

class GradientDiagnosticsHook(HookBase):
    """
    Hook to log gradient statistics for debugging mask collapse issues.
    
    Logs gradient norms for key layers like mask_embed to detect if
    gradients are vanishing or exploding.
    """
    
    def __init__(self, log_period: int = 100):
        self.log_period = log_period
    
    def after_step(self):
        """Log gradient statistics after backward pass."""
        if self.trainer.iter % self.log_period != 0:
            return
        
        if not comm.is_main_process():
            return
        
        model = self.trainer.model
        
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module
        
        print(f"\n=== GRADIENT DIAGNOSTICS (iter {self.trainer.iter}) ===")
        
        # Check mask head gradients
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # Focus on key layers
                if any(key in name for key in ['mask_embed', 'class_embed', 'decoder.layers']):
                    grad_stats[name] = {
                        'norm': grad_norm,
                        'mean': param.grad.mean().item(),
                        'max': param.grad.abs().max().item(),
                    }
        
        # Print most important gradients
        critical_layers = ['mask_embed', 'class_embed']
        for key in critical_layers:
            matching = [(n, s) for n, s in grad_stats.items() if key in n]
            if matching:
                for name, stats in matching[:2]:  # First 2 matches
                    print(f"  {name}:")
                    print(f"    Grad norm: {stats['norm']:.6f}, mean: {stats['mean']:.6f}, max: {stats['max']:.6f}")
        
        # Check for vanishing/exploding gradients
        all_norms = [s['norm'] for s in grad_stats.values() if s['norm'] > 0]
        if all_norms:
            avg_norm = sum(all_norms) / len(all_norms)
            max_norm = max(all_norms)
            min_norm = min(all_norms)
            print(f"  Avg grad norm: {avg_norm:.6f}, Min: {min_norm:.6f}, Max: {max_norm:.6f}")
            
            if max_norm > 100:
                print("  ⚠️  WARNING: Large gradients detected - consider gradient clipping")
            if min_norm < 1e-8:
                print("  ⚠️  WARNING: Very small gradients detected - possible vanishing gradient")


class NaNGradientSafetyHook(HookBase):
    """
    Lightweight hook to sanitize NaN/Inf gradients before optimizer step.
    
    NOTE: Actual gradient clipping is handled by detectron2's built-in
    maybe_add_gradient_clipping() which wraps the optimizer. We do NOT
    duplicate clipping here — that caused double clipping in previous runs,
    over-dampening gradients and preventing the model from learning.
    
    This hook ONLY handles NaN/Inf safety for AMP training.
    """
    def after_backward(self):
        """Sanitize NaN/Inf gradients (no clipping — that's in the optimizer)."""
        # Unscale gradients first if using AMP so we can inspect them
        if hasattr(self.trainer, 'grad_scaler') and self.trainer.grad_scaler is not None:
            try:
                self.trainer.grad_scaler.unscale_(self.trainer.optimizer)
            except RuntimeError:
                pass  # Already unscaled
        
        # NaN/Inf safety: zero out corrupted gradients to prevent accumulation
        nan_count = 0
        for p in self.trainer.model.parameters():
            if p.grad is not None:
                mask = torch.isnan(p.grad) | torch.isinf(p.grad)
                if mask.any():
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    nan_count += 1
        
        if nan_count > 0 and self.trainer.iter % 20 == 0:
            print(f"  [NaN Safety] Sanitized {nan_count} parameters with NaN/Inf grads at iter {self.trainer.iter}")

# ============================================================
# TRAINER
# ============================================================

class MultiViewTrainer(DefaultTrainer):
    """Trainer for multi-view Mask2Former."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Register NaN gradient safety hook (runs before optimizer step)
        # NOTE: Actual gradient clipping is handled by detectron2's built-in
        # maybe_add_gradient_clipping() which wraps the optimizer automatically
        # when CLIP_GRADIENTS.ENABLED=True. We do NOT add a separate clipping
        # hook — that caused DOUBLE CLIPPING in previous runs, severely
        # dampening gradient signal and preventing mask logit recovery.
        nan_safety_hook = NaNGradientSafetyHook()
        self._trainer.register_hooks([nan_safety_hook])
        logger.info(f"Registered NaN gradient safety hook")
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            logger.info(f"Gradient clipping via optimizer: type={cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE}, value={cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE}")

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(NaNLossCheckHook())
        # Add gradient diagnostics hook (every 100 iters)
        hooks.append(GradientDiagnosticsHook(log_period=100))
        # Add CSV logging hook
        hooks.append(CSVMetricsLogger(
            output_dir=self.cfg.OUTPUT_DIR,
            log_period=20,
        ))
        return hooks
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator for panoptic segmentation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOPanopticEvaluator(dataset_name, output_folder)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Run evaluation and log results to CSV.
        
        Overrides DefaultTrainer.test to capture evaluation results.
        """
        results = super().test(cfg, model, evaluators)
        
        # Log evaluation results to CSV
        if comm.is_main_process() and results:
            eval_csv_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_metrics.csv")
            
            # Check if file exists, if not create with headers
            if not os.path.exists(eval_csv_path):
                with open(eval_csv_path, 'w') as f:
                    headers = ['iteration', 'PQ', 'SQ', 'RQ', 'PQ_th', 'SQ_th', 'RQ_th', 'PQ_st', 'SQ_st', 'RQ_st']
                    f.write(','.join(headers) + '\n')
            
            # Extract panoptic metrics
            for dataset_name, dataset_results in results.items():
                if 'panoptic_seg' in dataset_results:
                    metrics = dataset_results['panoptic_seg']
                    
                    pq = metrics.get('PQ', 0.0)
                    sq = metrics.get('SQ', 0.0)
                    rq = metrics.get('RQ', 0.0)
                    pq_th = metrics.get('PQ_th', 0.0)
                    sq_th = metrics.get('SQ_th', 0.0)
                    rq_th = metrics.get('RQ_th', 0.0)
                    pq_st = metrics.get('PQ_st', 0.0)
                    sq_st = metrics.get('SQ_st', 0.0)
                    rq_st = metrics.get('RQ_st', 0.0)
                    
                    # Try to get current iteration (may not be available during eval_only)
                    try:
                        iteration = cfg.SOLVER.MAX_ITER  # Use as fallback
                    except Exception:
                        iteration = 0
                    
                    with open(eval_csv_path, 'a') as f:
                        values = [
                            str(iteration),
                            f'{pq:.4f}', f'{sq:.4f}', f'{rq:.4f}',
                            f'{pq_th:.4f}', f'{sq_th:.4f}', f'{rq_th:.4f}',
                            f'{pq_st:.4f}', f'{sq_st:.4f}', f'{rq_st:.4f}',
                        ]
                        f.write(','.join(values) + '\n')
                    
                    print(f"\n{'='*60}")
                    print(f"Panoptic Segmentation Evaluation Results:")
                    print(f"  PQ={pq:.2f}  SQ={sq:.2f}  RQ={rq:.2f}")
                    print(f"  PQ_th (things)={pq_th:.2f}  PQ_st (stuff)={pq_st:.2f}")
                    print(f"{'='*60}\n")
        
        return results
    
    @classmethod
    def run_view_analysis(cls, cfg, model):
        """
        Run detailed per-view analysis on evaluation dataset.
        
        This analyzes:
        - Per-view losses (reference vs targets)
        - Class probability consistency across views
        - Query prediction similarity
        
        Results are logged to view_analysis_metrics.csv
        """
        logger = logging.getLogger(__name__)
        logger.info("Running detailed view analysis...")
        
        # Build data loader
        data_loader = cls.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        
        # Create view analysis CSV
        output_dir = cfg.OUTPUT_DIR
        view_csv_path = os.path.join(output_dir, "view_analysis_metrics.csv")
        if not os.path.exists(view_csv_path):
            with open(view_csv_path, 'w') as f:
                headers = [
                    'iteration', 'scene_id', 'view_type', 'view_idx',
                    'loss_ce', 'loss_mask', 'loss_dice', 'total_loss',
                    'class_prob_kl', 'top_class_agreement'
                ]
                f.write(','.join(headers) + '\n')
        
        model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data_loader, desc="View Analysis")):
                if idx >= cfg.TEST.get('VIEW_ANALYSIS_MAX_SAMPLES', 100):
                    break
                
                # Get scene ID from batch
                scene_id = batch[0].get('scene_id', f'scene_{idx}')
                
                # Compute view metrics
                try:
                    view_metrics = model.compute_view_metrics(
                        batch[0],
                        compute_class_probs=True
                    )
                    
                    # Log to CSV
                    iteration = 0  # Can be updated if called during training
                    with open(view_csv_path, 'a') as f:
                        # Log reference view
                        ref_losses = view_metrics['ref_view_losses']
                        values = [
                            str(iteration),
                            str(scene_id),
                            'ref',
                            '0',
                            f"{ref_losses['loss_ce']:.6f}",
                            f"{ref_losses['loss_mask']:.6f}",
                            f"{ref_losses['loss_dice']:.6f}",
                            f"{sum(ref_losses.values()):.6f}",
                            '0.0',
                            '1.0',
                        ]
                        f.write(','.join(values) + '\n')
                        
                        # Log target views
                        target_losses = view_metrics['target_view_losses']
                        kl_divs = view_metrics.get('class_prob_kl_divergence', {}).get('per_view', [])
                        agreements = view_metrics.get('top_class_agreement', {}).get('per_view', [])
                        
                        for i, tgt_loss in enumerate(target_losses):
                            values = [
                                str(iteration),
                                str(scene_id),
                                'target',
                                str(tgt_loss['view_idx']),
                                f"{tgt_loss['loss_ce']:.6f}",
                                f"{tgt_loss['loss_mask']:.6f}",
                                f"{tgt_loss['loss_dice']:.6f}",
                                f"{tgt_loss['total']:.6f}",
                                f"{kl_divs[i] if i < len(kl_divs) else 0.0:.6f}",
                                f"{agreements[i] if i < len(agreements) else 0.0:.6f}",
                            ]
                            f.write(','.join(values) + '\n')
                    
                    # Print summary
                    if idx % 10 == 0:
                        kl_mean = view_metrics.get('class_prob_kl_divergence', {}).get('mean', 0.0)
                        agree_mean = view_metrics.get('top_class_agreement', {}).get('mean', 0.0)
                        logger.info(
                            f"Scene {scene_id}: "
                            f"KL div={kl_mean:.4f}, "
                            f"Class agreement={agree_mean:.4f}"
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze scene {scene_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        logger.info(f"View analysis complete. Results saved to {view_csv_path}")
        
        return view_csv_path
    
    def after_train(self):
        """Close TensorBoard writer after training completes."""
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
                logger.info("TensorBoard writer closed successfully")
            except Exception as e:
                print(f"Warning: Failed to close TensorBoard writer: {e}")
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with multi-view collation."""
        from detectron2.data import DatasetCatalog
        from detectron2.data.samplers import TrainingSampler
        from detectron2.data.common import DatasetFromList, MapDataset
        from detectron2.data.build import trivial_batch_collator
        from torch.utils.data import DataLoader
        
        from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
            ScanNetPPMultiViewDatasetMapper
        )
        
        # Get dataset
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        # Filter out scenes with no camera data by checking early
        # (This pre-filters based on scene structure, actual None filtering happens in mapper)
        logger.info(f"Loaded {len(dataset_dicts)} dataset entries")
        
        mapper = ScanNetPPMultiViewDatasetMapper(cfg, is_train=True)
        
        # Create dataset with mapper
        dataset = DatasetFromList(dataset_dicts, copy=False)
        dataset = MapDataset(dataset, mapper)
        
        # Use module-level FilteredDataset (picklable for spawn multiprocessing)
        filtered_dataset = FilteredDataset(dataset)
        
        # Create sampler and loader
        sampler = TrainingSampler(len(dataset))
        
        return DataLoader(
            filtered_dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=multi_view_collate_fn,
            worker_init_fn=dataloader_worker_init_fn,
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test loader."""
        from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
            ScanNetPPMultiViewDatasetMapper
        )
        
        mapper = ScanNetPPMultiViewDatasetMapper(cfg, is_train=False)
        
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=mapper,
            collate_fn=multi_view_collate_fn,
        )


def register_scannetpp_datasets(args):
    """
    Register ScanNet++ datasets for training and evaluation.
    
    This function registers the datasets specified via command line args.
    Users must provide:
      --scannetpp-root: Path to ScanNet++ data directory
      --panoptic-root: Path to rasterized panoptic annotations
      --split-dir: Directory containing split files (optional, defaults to scannetpp-root/splits)
    
    Alternatively, if these args are not provided, users should register datasets
    manually before calling this script.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # Check if custom registration is needed
    if not hasattr(args, 'scannetpp_root') or args.scannetpp_root is None:
        logger.info("No --scannetpp-root provided. Assuming datasets are pre-registered.")
        return
    
    from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
        register_scannetpp_panoptic
    )
    
    scannetpp_root = args.scannetpp_root
    panoptic_root = args.panoptic_root or os.path.join(os.path.dirname(scannetpp_root), "panoptic_annotations")
    split_dir = args.split_dir or os.path.join(os.path.dirname(scannetpp_root), "splits")
    
    num_classes = getattr(args, 'num_classes', 1000)
    image_type = getattr(args, 'image_type', 'dslr')
    use_undistorted = getattr(args, 'use_undistorted', True)
    
    # Register training dataset
    train_split = os.path.join(split_dir, "nvs_sem_train.txt")
    if os.path.exists(train_split):
        logger.info(f"Registering training dataset from {train_split}")
        try:
            register_scannetpp_panoptic(
                name="scannetpp_panoptic_train",
                data_root=scannetpp_root,
                split_file=train_split,
                panoptic_dir=panoptic_root,
                image_type=image_type,
                use_undistorted=use_undistorted,
                num_classes=num_classes,
            )
            logger.info("✓ Registered: scannetpp_panoptic_train")
        except Exception as e:
            logger.warning(f"Failed to register training dataset: {e}")
    else:
        logger.warning(f"Training split not found: {train_split}")
    
    # Register validation dataset
    val_split = os.path.join(split_dir, "nvs_sem_val.txt")
    if os.path.exists(val_split):
        logger.info(f"Registering validation dataset from {val_split}")
        try:
            register_scannetpp_panoptic(
                name="scannetpp_panoptic_val",
                data_root=scannetpp_root,
                split_file=val_split,
                panoptic_dir=panoptic_root,
                image_type=image_type,
                use_undistorted=use_undistorted,
                num_classes=num_classes,
            )
            logger.info("✓ Registered: scannetpp_panoptic_val")
        except Exception as e:
            logger.warning(f"Failed to register validation dataset: {e}")
    else:
        logger.warning(f"Validation split not found: {val_split}")


def main(args):
    """Main training function."""
    # Register datasets FIRST (using the standalone registration script)
    try:
        from register_datasets import register_scannetpp_train_val
        logger.info("Registering ScanNet++ datasets...")
        register_scannetpp_train_val()
    except ImportError as e:
        logger.warning(f"Could not import register_datasets: {e}")
        logger.info("Attempting inline registration...")
        register_scannetpp_datasets(args)
    except Exception as e:
        logger.error(f"Dataset registration failed: {e}")
        # Try fallback registration
        register_scannetpp_datasets(args)

    cfg = setup_cfg(args)
    
    if args.eval_only:
        model = MultiViewTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
        # Run view analysis if requested
        if hasattr(args, 'view_analysis') and args.view_analysis:
            res = MultiViewTrainer.run_view_analysis(cfg, model)
        else:
            res = MultiViewTrainer.test(cfg, model)
        return res
    
    trainer = MultiViewTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    
    # View analysis mode
    parser.add_argument(
        "--view-analysis",
        action="store_true",
        help="Run detailed per-view analysis during evaluation"
    )
    
    # ========================================
    # PANOPTIC LABEL GENERATION
    # ========================================
    parser.add_argument(
        "--generate-labels",
        action="store_true",
        help="Generate 2D panoptic labels from ScanNet++ 3D mesh before training"
    )
    parser.add_argument(
        "--generate-labels-only",
        action="store_true",
        help="Only generate labels, don't start training"
    )
    parser.add_argument(
        "--label-gen-workers",
        type=int,
        default=4,
        help="Number of parallel workers for label generation"
    )
    
    # ========================================
    # DATASET PATHS
    # ========================================
    parser.add_argument(
        "--scannetpp-root",
        type=str,
        default=None,
        help="Path to ScanNet++ data directory (contains scene folders)"
    )
    parser.add_argument(
        "--panoptic-root",
        type=str,
        default=None,
        help="Path to rasterized panoptic annotations directory"
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Path to directory containing split files (nvs_sem_train.txt, etc.)"
    )
    
    # ========================================
    # DATASET OPTIONS
    # ========================================
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of semantic classes in the dataset (ScanNet++ has 1000 classes)"
    )
    parser.add_argument(
        "--image-type",
        type=str,
        default="dslr",
        choices=["dslr", "iphone"],
        help="Image type to use (dslr or iphone)"
    )
    parser.add_argument(
        "--use-undistorted",
        action="store_true",
        default=True,
        help="Use undistorted images (default: True)"
    )
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
