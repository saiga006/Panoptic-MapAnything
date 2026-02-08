"""
Inference script for Mask2Former with MapAnything backbone

Supports TWO inference modes:

1. SINGLE-VIEW INFERENCE (Original):
   - 3D reconstruction (depth + point cloud) from MapAnything
   - Panoptic segmentation from trained Mask2Former

2. MULTI-VIEW INFERENCE (NEW):
   - Feed multiple views of same scene to MapAnything
   - Use Query Propagation for consistent instance IDs across views
   - Generate unified 3D panoptic point cloud with consistent labels
   
Multi-View Pipeline:
    N Views → MapAnything → 3D Points + Depth per view
         ↓
    Mask2Former with Query Propagation
         ↓
    Lift 2D predictions to 3D (per view)
         ↓
    Fuse into unified Panoptic Point Cloud
    
Key Insight: Query #K in ALL views = Same physical object instance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T

# MapAnything imports
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Add Mask2Former to path
import sys
import os
# Robustly add paths relative to this script
script_path = Path(__file__).resolve()
script_dir = script_path.parent
sys.path.insert(0, str(script_dir))
# Add ScanNet++ utils path: Mask2Former/configs/scannetpp/scannetpp
scannetpp_utils_path = script_dir / 'configs' / 'scannetpp' / 'scannetpp'
sys.path.insert(0, str(scannetpp_utils_path))

from mask2former import add_maskformer2_config

# Import the custom backbone from training script
# This registers the MapAnythingWithPanopticDPT with Detectron2
# from m2f_train_cluster_working import MapAnythingWithPanopticDPT, build_mapanything_panoptic_dpt_backbone


# ============================================================
# PANOPTIC POINT CLOUD DATA STRUCTURE
# ============================================================

@dataclass
class PanopticPointCloud:
    """Output container for 3D panoptic segmentation."""
    points: np.ndarray              # [P, 3] - 3D coordinates in world frame
    instance_ids: np.ndarray        # [P] - instance ID (corresponds to query index)
    semantic_classes: np.ndarray    # [P] - semantic class ID
    confidences: np.ndarray         # [P] - prediction confidence
    colors: Optional[np.ndarray] = None  # [P, 3] - RGB colors (0-1 range)
    
    def save_ply(self, path: str, use_semantic_colors: bool = False, 
                 class_colors: Optional[np.ndarray] = None):
        """
        Save panoptic point cloud as PLY file with custom properties.
        
        Args:
            path: Output file path
            use_semantic_colors: If True, color by semantic class instead of RGB
            class_colors: Optional [num_classes, 3] array of class colors
        """
        with open(path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property int instance_id\n")
            f.write("property int semantic_class\n")
            f.write("property float confidence\n")
            f.write("end_header\n")
            
            # Determine colors
            if use_semantic_colors and class_colors is not None:
                num_classes = len(class_colors)
                colors_to_use = class_colors[np.clip(self.semantic_classes, 0, num_classes - 1)]
            elif self.colors is not None:
                colors_to_use = (self.colors * 255).astype(np.uint8)
            else:
                colors_to_use = np.full((len(self.points), 3), 128, dtype=np.uint8)
            
            # Write data
            for i in range(len(self.points)):
                f.write(f"{self.points[i, 0]:.6f} {self.points[i, 1]:.6f} {self.points[i, 2]:.6f} "
                       f"{int(colors_to_use[i, 0])} {int(colors_to_use[i, 1])} {int(colors_to_use[i, 2])} "
                       f"{int(self.instance_ids[i])} {int(self.semantic_classes[i])} "
                       f"{self.confidences[i]:.4f}\n")
        
        print(f"Saved panoptic point cloud to {path}")
        print(f"  Points: {len(self.points)}")
        print(f"  Unique instances: {len(np.unique(self.instance_ids))}")
        print(f"  Unique classes: {len(np.unique(self.semantic_classes))}")
    
    def to_open3d(self, use_semantic_colors: bool = False,
                  class_colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """Convert to Open3D point cloud for visualization."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        if use_semantic_colors and class_colors is not None:
            num_classes = len(class_colors)
            colors = class_colors[np.clip(self.semantic_classes, 0, num_classes - 1)] / 255.0
        elif self.colors is not None:
            colors = self.colors
        else:
            colors = np.full((len(self.points), 3), 0.5)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


# ============================================================
# MULTI-VIEW PANOPTIC INFERENCE
# ============================================================

class MultiViewPanopticInference:
    """
    Multi-View Inference for generating panoptic 3D point clouds.
    
    Key insight: By using query propagation, Query #K in all views
    represents the SAME physical object. This gives us consistent
    instance IDs across views without any post-processing!
    
    Pipeline:
    1. Feed N views to MapAnything → get depth, 3D points per view
    2. Run Mask2Former on reference view → get refined queries
    3. Propagate queries to target views (same head, same queries)
    4. Lift 2D masks to 3D points per view
    5. Fuse all views into unified panoptic point cloud
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        voxel_size: float = 0.02,  # 2cm voxels for deduplication
        num_classes: int = 133,
    ):
        """
        Args:
            model: Trained MultiViewMask2Former model
            device: Device to run inference on
            confidence_threshold: Minimum confidence to include a point
            mask_threshold: Threshold for binary mask prediction
            voxel_size: Voxel size for point cloud deduplication
            num_classes: Number of semantic classes
        """
        self.model = model.to(device).eval()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        
        # Generate class colors
        np.random.seed(42)
        self.class_colors = np.random.randint(0, 255, size=(num_classes, 3))
    
    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ) -> Tuple[PanopticPointCloud, List[Dict[str, torch.Tensor]]]:
        """
        Generate panoptic 3D point cloud from multiple views.
        
        Args:
            images: [N, 3, H, W] - N views of the same scene
            camera_poses: [N, 4, 4] - camera-to-world transforms
            camera_intrinsics: [N, 3, 3] - camera intrinsic matrices
            depths: [N, H, W] - optional depth maps (if None, use MapAnything)
        
        Returns:
            PanopticPointCloud with fused 3D predictions
            List of per-view 2D prediction dicts (pred_masks, pred_classes, pred_scores)
        """
        N, C, H, W = images.shape
        
        # Move to device
        images = images.to(self.device).float()
        camera_poses = camera_poses.to(self.device)
        camera_intrinsics = camera_intrinsics.to(self.device)
        if depths is not None:
            depths = depths.to(self.device)
        
        # NOTE: NO MANUAL NORMALIZATION NEEDED!
        # The MapAnythingMultiViewBackbone.forward() passes 'data_norm_type': ['dinov2'] to MapAnything,
        # which tells it to normalize internally using DINOv2/ImageNet stats.
        # Training uses 0-255 images directly from dataset mapper → backbone handles normalization.
        # If we pre-normalize here, we get DOUBLE NORMALIZATION WHICH DESTROYS THE FEATURES.
        print(f"[DEBUG] Input images stats: Min={images.min():.3f} Max={images.max():.3f}")
        
        # Ensure images are in 0-255 range (matching training data pipeline)
        if images.max() <= 1.0:
            print("[DEBUG] Images appear to be in 0-1 range, scaling to 0-255")
            images = images * 255.0
        
        print(f"[DEBUG] Images for backbone (should be 0-255): Min={images.min():.3f} Max={images.max():.3f}")

        # Add batch dimension [1, N, 3, H, W]
        images_batched = images.unsqueeze(0)

        camera_poses_batched = camera_poses.unsqueeze(0)
        camera_intrinsics_batched = camera_intrinsics.unsqueeze(0)
        
        print(f"Processing {N} views...")
        
        # Step 1: Run backbone to get features + depth
        print("  Running MapAnything backbone...")
        backbone_outputs = self.model.backbone(
            images=images_batched,
            camera_poses=camera_poses_batched,
            camera_intrinsics=camera_intrinsics_batched,
            return_all_views=True,
        )
        
        all_view_features = backbone_outputs['all_view_features']
        
        # Debug: Check backbone feature statistics
        print("\n[DEBUG] Backbone Feature Statistics:")
        for key in all_view_features:
            feat = all_view_features[key] # [N, B, C, H, W]
            print(f"  {key}: shape={feat.shape}, min={feat.min():.3f}, max={feat.max():.3f}, mean={feat.mean():.3f}, std={feat.std():.3f}")

        # Get depth from MapAnything if not provided
        if depths is None:
            print("  Extracting depth from MapAnything...")
            depths, confidences = self._get_depths_from_mapanything(
                images_batched, camera_poses_batched
            )
        else:
            confidences = torch.ones(N, H, W, device=self.device)
        
        # Step 2: Run Mask2Former with query propagation
        print("  Running Mask2Former with query propagation...")
        per_view_predictions = self._run_mask2former_all_views(
            all_view_features=all_view_features,
            depths=depths,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            N=N,
        )
        
        # Step 3: Lift 2D predictions to 3D per view
        print("  Lifting 2D predictions to 3D...")
        all_points = []
        all_instance_ids = []
        all_classes = []
        all_confs = []
        all_colors = []
        
        for view_idx in range(N):
            pred = per_view_predictions[view_idx]
            depth = depths[view_idx]  # [H, W]
            conf = confidences[view_idx]  # [H, W]
            pose = camera_poses[view_idx]  # [4, 4]
            K = camera_intrinsics[view_idx]  # [3, 3]
            img = images[view_idx]  # [3, H, W]
            
            points, inst_ids, classes, point_confs, colors = self._lift_to_3d(
                pred_masks=pred['pred_masks'],      # [Q, H, W]
                pred_classes=pred['pred_classes'],  # [Q]
                pred_scores=pred['pred_scores'],    # [Q]
                depth=depth,
                confidence=conf,
                pose=pose,
                intrinsics=K,
                image=img,
            )
            
            if len(points) > 0:
                all_points.append(points)
                all_instance_ids.append(inst_ids)
                all_classes.append(classes)
                all_confs.append(point_confs)
                all_colors.append(colors)
                print(f"    View {view_idx}: {len(points)} points")
        
        # Step 4: Fuse multi-view predictions
        print("  Fusing multi-view predictions...")
        fused = self._fuse_multiview_points(
            all_points, all_instance_ids, all_classes, all_confs, all_colors
        )
        
        print(f"✓ Generated panoptic point cloud: {len(fused.points)} points")
        
        return fused, per_view_predictions
    
    def _get_depths_from_mapanything(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get depth predictions from MapAnything.
        
        Args:
            images: [1, N, 3, H, W]
            camera_poses: [1, N, 4, 4]
        
        Returns:
            depths: [N, H, W]
            confidences: [N, H, W]
        """
        B, N, C, H, W = images.shape
        mapanything = self.model.backbone.mapanything
        
        # Prepare views for MapAnything
        views = []
        for v in range(N):
            view_dict = {
                'img': images[:, v],  # [B, C, H, W]
                'data_norm_type': ['dinov2'] * B,
            }
            views.append(view_dict)
        
        # Run MapAnything forward
        with torch.no_grad():
            outputs = mapanything(views)
        
        # Extract depth and confidence
        depths = []
        confidences = []
        
        for v in range(N):
            pred = outputs[v]
            
            if 'depth_along_ray' in pred:
                depth = pred['depth_along_ray'].squeeze()
                if 'metric_scaling_factor' in pred:
                    scale = pred['metric_scaling_factor']
                    if isinstance(scale, torch.Tensor):
                        scale = scale.item()
                    depth = depth * scale
            elif 'pts3d' in pred:
                # Compute depth from 3D points (z-coordinate in camera frame)
                pts3d = pred['pts3d'].squeeze()  # [H, W, 3]
                depth = pts3d[..., 2]
            else:
                # Fallback: uniform depth
                depth = torch.ones(H, W, device=self.device)
            
            depths.append(depth)
            
            if 'conf' in pred:
                conf = pred['conf'].squeeze()
            else:
                conf = torch.ones_like(depth)
            confidences.append(conf)
        
        depths = torch.stack(depths, dim=0)  # [N, H, W]
        confidences = torch.stack(confidences, dim=0)  # [N, H, W]
        
        return depths, confidences
    
    def _prepare_features_for_view(
        self,
        all_view_features: Dict[str, torch.Tensor],
        view_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract features for a specific view."""
        return {
            key: feats[view_idx]  # [B, C, H', W']
            for key, feats in all_view_features.items()
        }
    
    def _run_pixel_decoder(self, features: Dict[str, torch.Tensor]):
        """Run pixel decoder on features."""
        mask_features, transformer_encoder_features, multi_scale_features = \
            self.model.sem_seg_head.pixel_decoder.forward_features(features)
        return mask_features, multi_scale_features
    
    def _run_mask2former_all_views(
        self,
        all_view_features: Dict[str, torch.Tensor],
        depths: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        N: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run Mask2Former with query propagation across all views.
        
        Returns list of predictions, one per view, with CONSISTENT query IDs.
        """
        predictions = [None] * N  # Pre-allocate to maintain order
        
        # Select reference view (first view or could be more sophisticated)
        ref_idx = 0
        
        # Run reference view with learnable queries
        ref_features = self._prepare_features_for_view(all_view_features, ref_idx)
        ref_mask_features, ref_multi_scale_features = self._run_pixel_decoder(ref_features)
        
        ref_outputs = self.model.query_propagation_decoder(
            x=ref_multi_scale_features,
            mask_features=ref_mask_features,
            initial_query_feat=None,  # Use learnable queries for reference
            initial_attn_mask=None,
        )
        
        # Get refined queries for propagation
        query_embeddings = ref_outputs['query_embeddings']  # [Q, 1, D]
        
        # Process reference view predictions
        ref_pred = self._process_predictions(ref_outputs)
        predictions[ref_idx] = ref_pred
        
        # Store reference masks for warping (used for spatial bridging)
        ref_pred_masks = ref_outputs['pred_masks']  # [1, Q, H', W']
        ref_depth = depths[ref_idx:ref_idx+1].unsqueeze(1)  # [1, 1, H, W]
        ref_pose = camera_poses[ref_idx:ref_idx+1]  # [1, 4, 4]
        
        # Run target views with propagated queries
        for tgt_idx in range(N):
            if tgt_idx == ref_idx:
                continue
            
            # Get target view features
            tgt_features = self._prepare_features_for_view(all_view_features, tgt_idx)
            tgt_mask_features, tgt_multi_scale_features = self._run_pixel_decoder(tgt_features)
            
            # Create warped attention mask for spatial bridging
            tgt_pose = camera_poses[tgt_idx:tgt_idx+1]  # [1, 4, 4]
            K = camera_intrinsics[tgt_idx:tgt_idx+1]    # [1, 3, 3]
            tgt_depth = depths[tgt_idx:tgt_idx+1].unsqueeze(1) # [1, 1, H, W]
            
            warped_attn_mask = self._create_warped_attention_mask(
                ref_pred_masks=ref_pred_masks,
                ref_depth=ref_depth,
                ref_pose=ref_pose,
                tgt_pose=tgt_pose,
                intrinsics=K,
                target_size=tgt_multi_scale_features[0].shape[-2:],
                tgt_depth=tgt_depth,
            )
            
# Alpha blending of warped/predicted masks happens internally in decoder
# (gradual transition: warped → learned across layers)
            tgt_outputs = self.model.query_propagation_decoder(
                x=tgt_multi_scale_features,
                mask_features=tgt_mask_features,
                initial_query_feat=query_embeddings,  # Propagate reference queries
                initial_attn_mask=warped_attn_mask,   # Spatial guidance
            )
            
            tgt_pred = self._process_predictions(tgt_outputs)
            predictions[tgt_idx] = tgt_pred
        
        return predictions
    
    def _create_warped_attention_mask(
        self,
        ref_pred_masks: torch.Tensor,
        ref_depth: torch.Tensor,
        ref_pose: torch.Tensor,
        tgt_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        target_size: Tuple[int, int],
        tgt_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create warped attention mask for spatial bridging."""
        # Import the warping function from training script
        from m2f_train_multiview import create_warped_attention_mask
        
        return create_warped_attention_mask(
            ref_pred_masks=ref_pred_masks,
            ref_depth=ref_depth,
            ref_pose=ref_pose,
            tgt_pose=tgt_pose,
            intrinsics=intrinsics,
            target_size=target_size,
            tgt_depth=tgt_depth,
            mask_threshold=self.mask_threshold,
        )
    
    def _process_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Process raw Mask2Former outputs into final predictions."""
        pred_logits = outputs['pred_logits'][0]  # [Q, C+1]
        pred_masks_logits = outputs['pred_masks'][0]    # [Q, H', W'] - raw logits!
        
        # CRITICAL: Apply sigmoid to convert mask logits to probabilities!
        pred_masks = torch.sigmoid(pred_masks_logits)  # [Q, H', W'] in [0, 1]
        
        # Get class predictions (excluding no-object class)
        scores = F.softmax(pred_logits, dim=-1)[:, :-1]  # [Q, C]
        pred_scores, pred_classes = scores.max(dim=-1)   # [Q], [Q]
        
        # Debug: Check outputs
        print(f"[DEBUG] Processing predictions:")
        print(f"  Logits range: {pred_logits.min():.3f} - {pred_logits.max():.3f}")
        print(f"  Mask logits range: {pred_masks_logits.min():.3f} - {pred_masks_logits.max():.3f}")
        print(f"  Masks range (after sigmoid): {pred_masks.min():.3f} - {pred_masks.max():.3f}")
        print(f"  Scores range (softmax max): {pred_scores.min():.3f} - {pred_scores.max():.3f}")
        print(f"  Classes dist: {torch.unique(pred_classes, return_counts=True)}")

        return {
            'pred_masks': pred_masks,      # [Q, H, W] - probabilities in [0, 1]
            'pred_classes': pred_classes,  # [Q]
            'pred_scores': pred_scores,    # [Q]
        }
    
    def _lift_to_3d(
        self,
        pred_masks: torch.Tensor,     # [Q, H', W']
        pred_classes: torch.Tensor,   # [Q]
        pred_scores: torch.Tensor,    # [Q]
        depth: torch.Tensor,          # [H, W]
        confidence: torch.Tensor,     # [H, W]
        pose: torch.Tensor,           # [4, 4]
        intrinsics: torch.Tensor,     # [3, 3]
        image: torch.Tensor,          # [3, H, W]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Lift 2D predictions to 3D points.
        
        Key insight: Each query index IS the instance ID.
        Query #K in all views → Instance #K in 3D.
        """
        Q = pred_masks.shape[0]
        H, W = depth.shape
        device = pred_masks.device
        
        # Resize masks to match depth resolution
        if pred_masks.shape[-2:] != (H, W):
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )[0]
        
        # Create pixel grid
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Valid depth mask
        valid_depth = (depth > 0) & (confidence > self.confidence_threshold)
        
        # Debug stats
        if valid_depth.sum() == 0:
            print(f"[DEBUG] No valid depth points!")
            print(f"  Depth range: {depth.min():.3f} - {depth.max():.3f}")
            print(f"  Conf range: {confidence.min():.3f} - {confidence.max():.3f} (Threshold: {self.confidence_threshold})")
            print(f"  Depth > 0 count: {(depth > 0).sum()}")
            print(f"  Conf > th count: {(confidence > self.confidence_threshold).sum()}")
        
        # Unproject to 3D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam = (u_coords - cx) * depth / fx
        y_cam = (v_coords - cy) * depth / fy
        z_cam = depth
        
        # Stack to [H, W, 4] homogeneous
        pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
        
        # Transform to world [H, W, 4]
        pts_world = torch.einsum('hwi,ji->hwj', pts_cam, pose)
        pts_world = pts_world[..., :3]  # [H, W, 3]
        
        # Assign each pixel to its best query (instance)
        # [Q, H, W] -> find argmax query per pixel
        weighted_masks = pred_masks * pred_scores.view(Q, 1, 1)
        instance_ids = weighted_masks.argmax(dim=0)  # [H, W] - query index = instance ID
        
        # Get semantic class for each pixel (from assigned query)
        semantic_classes = pred_classes[instance_ids]  # [H, W]
        
        # Get confidence for each pixel
        max_mask_score = weighted_masks.max(dim=0)[0]  # [H, W]
        point_confs = max_mask_score * confidence  # Combine mask and depth confidence
        
        # Filter valid points
        valid = valid_depth & (max_mask_score > self.mask_threshold)

        if valid.sum() == 0 and valid_depth.sum() > 0:
            print(f"[DEBUG] Depth is valid but Masks filtered everything!")
            print(f"  Mask score range: {max_mask_score.min():.3f} - {max_mask_score.max():.3f} (Threshold: {self.mask_threshold})")
            print(f"  Max valid mask score: {max_mask_score[valid_depth].max() if valid_depth.any() else 'N/A'}")
        
        # Extract valid points
        points = pts_world[valid].cpu().numpy()           # [P, 3]
        inst_ids = instance_ids[valid].cpu().numpy()      # [P]
        classes = semantic_classes[valid].cpu().numpy()   # [P]
        confs = point_confs[valid].cpu().numpy()          # [P]
        
        # Get colors (image already in [0, 1] range from preprocessing)
        img_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        colors = img_np.reshape(H, W, 3)[valid.cpu().numpy()]  # [P, 3]
        
        return points, inst_ids, classes, confs, colors
    
    def _fuse_multiview_points(
        self,
        all_points: List[np.ndarray],
        all_instance_ids: List[np.ndarray],
        all_classes: List[np.ndarray],
        all_confs: List[np.ndarray],
        all_colors: List[np.ndarray],
    ) -> PanopticPointCloud:
        """
        Fuse points from multiple views into unified point cloud.
        
        Key: Instance IDs are ALREADY consistent because we used
        query propagation. Query #K in View A = Query #K in View B.
        """
        if len(all_points) == 0:
            return PanopticPointCloud(
                points=np.zeros((0, 3)),
                instance_ids=np.zeros(0, dtype=np.int32),
                semantic_classes=np.zeros(0, dtype=np.int32),
                confidences=np.zeros(0),
                colors=np.zeros((0, 3)),
            )
        
        # Concatenate all points
        points = np.concatenate(all_points, axis=0)
        instance_ids = np.concatenate(all_instance_ids, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        confs = np.concatenate(all_confs, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        
        # Voxelize to remove duplicates and vote on labels
        if self.voxel_size > 0 and len(points) > 0:
            points, instance_ids, classes, confs, colors = self._voxelize_and_vote(
                points, instance_ids, classes, confs, colors
            )
        
        return PanopticPointCloud(
            points=points,
            instance_ids=instance_ids.astype(np.int32),
            semantic_classes=classes.astype(np.int32),
            confidences=confs,
            colors=colors,
        )
    
    def _voxelize_and_vote(
        self,
        points: np.ndarray,
        instance_ids: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize point cloud and vote on instance/class labels.
        
        For overlapping points from different views, we:
        1. Group by voxel
        2. Average positions
        3. Vote on instance ID (weighted by confidence)
        4. Vote on class (weighted by confidence)
        """
        if len(points) == 0:
            return points, instance_ids, classes, confs, colors
        
        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int64)
        
        # Shift to positive
        voxel_min = voxel_indices.min(axis=0)
        voxel_indices = voxel_indices - voxel_min
        voxel_max = voxel_indices.max(axis=0) + 1
        
        # Create unique voxel keys
        voxel_keys = (
            voxel_indices[:, 0] * voxel_max[1] * voxel_max[2] +
            voxel_indices[:, 1] * voxel_max[2] +
            voxel_indices[:, 2]
        )
        
        # Find unique voxels
        unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
        num_voxels = len(unique_keys)
        
        # Aggregate per voxel using vectorized operations
        fused_points = np.zeros((num_voxels, 3))
        fused_instance_ids = np.zeros(num_voxels, dtype=np.int64)
        fused_classes = np.zeros(num_voxels, dtype=np.int64)
        fused_confs = np.zeros(num_voxels)
        fused_colors = np.zeros((num_voxels, 3))
        
        # Use np.bincount for efficient aggregation
        for dim in range(3):
            weighted_sum = np.bincount(inverse_indices, weights=points[:, dim] * confs, minlength=num_voxels)
            weight_sum = np.bincount(inverse_indices, weights=confs, minlength=num_voxels)
            weight_sum = np.maximum(weight_sum, 1e-8)  # Avoid division by zero
            fused_points[:, dim] = weighted_sum / weight_sum
        
        for dim in range(3):
            weighted_sum = np.bincount(inverse_indices, weights=colors[:, dim] * confs, minlength=num_voxels)
            weight_sum = np.bincount(inverse_indices, weights=confs, minlength=num_voxels)
            weight_sum = np.maximum(weight_sum, 1e-8)
            fused_colors[:, dim] = weighted_sum / weight_sum
        
        # For instance and class, take the one with highest total confidence
        for v_idx in range(num_voxels):
            mask = inverse_indices == v_idx
            voxel_confs = confs[mask]
            
            # Instance voting
            voxel_instances = instance_ids[mask]
            unique_instances = np.unique(voxel_instances)
            best_instance = unique_instances[0]
            best_conf = 0
            for inst in unique_instances:
                inst_conf = voxel_confs[voxel_instances == inst].sum()
                if inst_conf > best_conf:
                    best_conf = inst_conf
                    best_instance = inst
            fused_instance_ids[v_idx] = best_instance
            
            # Class voting
            voxel_classes = classes[mask]
            unique_classes_v = np.unique(voxel_classes)
            best_class = unique_classes_v[0]
            best_conf = 0
            for cls in unique_classes_v:
                cls_conf = voxel_confs[voxel_classes == cls].sum()
                if cls_conf > best_conf:
                    best_conf = cls_conf
                    best_class = cls
            fused_classes[v_idx] = best_class
            
            # Max confidence
            fused_confs[v_idx] = voxel_confs.max()
        
        return fused_points, fused_instance_ids, fused_classes, fused_confs, fused_colors


# ============================================================
# MULTI-VIEW INFERENCE UTILITIES
# ============================================================

def load_multiview_model(checkpoint_path: str, config_file: Optional[str] = None):
    """
    Load trained MultiViewMask2Former model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_file: Optional path to config file
    
    Returns:
        model: Loaded model in eval mode
        cfg: Configuration object
    """
    from m2f_train_multiview import MultiViewMask2Former, setup_cfg, add_multiview_config
    
    # Create a minimal args object
    class Args:
        def __init__(self):
            self.config_file = config_file
            self.opts = []
    
    args = Args()
    cfg = setup_cfg(args)
    
    # Build model
    model = MultiViewMask2Former(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ Loaded multi-view model from: {checkpoint_path}")
    
    return model, cfg


def load_depth_map(image_path: Path, scene_path: Path) -> Optional[np.ndarray]:
    """
    Load ground truth depth map from ScanNet++ structure.
    Checks:
    1. dslr/undistorted_render_depth/{stem}.png
    2. dslr/render_depth/{stem}.png
    
    Returns:
        Depth map in meters (float32), or None if not found.
    """
    image_stem = image_path.stem
    
    # Try undistorted depth first (matching scannetpp_panoptic_dataset_mapper)
    possible_paths = [
        scene_path / "dslr" / "undistorted_render_depth" / f"{image_stem}.png",
        scene_path / "dslr" / "render_depth" / f"{image_stem}.png",
        # Check beside image (generic structure)
        image_path.parent.parent / "undistorted_render_depth" / f"{image_stem}.png",
        image_path.parent.parent / "render_depth" / f"{image_stem}.png",
        # Fallback to same folder (for general datasets)
        image_path.parent / f"{image_stem}.png",
    ]
    
    depth_path = None
    for p in possible_paths:
        if p.exists():
            depth_path = p
            break
            
    if depth_path is None:
        return None
        
    try:
        # Load uint16 PNG depth (mm)
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            return None
        
        # Convert mm to meters
        depth_m = depth_mm.astype(np.float32) / 1000.0
        return depth_m
    except Exception as e:
        print(f"Error loading depth {depth_path}: {e}")
        return None


# ============================================================
# OVERLAP-AWARE VIEW SELECTION (ported from training mapper)
# ============================================================

def _compute_camera_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """Compute Euclidean distance between camera centers."""
    center1 = pose1[:3, 3]
    center2 = pose2[:3, 3]
    return np.linalg.norm(center1 - center2)


def _compute_viewing_direction_similarity(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """Compute cosine similarity between camera viewing directions.
    Viewing direction is the negative Z-axis of the camera frame.
    Returns: Cosine similarity in [-1, 1], where 1 = same direction"""
    dir1 = -pose1[:3, 2]
    dir2 = -pose2[:3, 2]
    return np.dot(dir1, dir2)


def _select_diverse_views(
    poses: List[np.ndarray],
    num_views: int,
    min_distance: float = 0.3,
    max_distance: float = 2.0,
    seed: Optional[int] = None,
) -> List[int]:
    """Select views with HIGH visual overlap for query propagation.
    
    For multi-view inference with query propagation, views need significant
    overlap (50-70%) so that warped attention masks are meaningful.
    
    Algorithm:
    1. Start with a random reference view
    2. Score candidates by: close distance + similar viewing direction
    3. Prefer nearby views looking at similar regions (high overlap)
    
    Args:
        poses: List of 4x4 camera-to-world matrices (numpy)
        num_views: Number of views to select
        min_distance: Minimum distance (meters) between views (avoid identical)
        max_distance: Maximum distance (meters) — views beyond this have low overlap
        seed: Random seed for reproducibility (None = random each call)
    
    Returns:
        List of indices of selected views (index 0 = reference)
    """
    if len(poses) <= num_views:
        return list(range(len(poses)))
    
    rng = np.random.default_rng(seed)
    n = len(poses)
    
    # Start with a random reference view
    ref_idx = rng.integers(n)
    selected = [ref_idx]
    
    # Precompute scores relative to reference
    scores = np.full(n, -np.inf)
    for i in range(n):
        if i == ref_idx:
            continue
        dist = _compute_camera_distance(poses[i], poses[ref_idx])
        dir_sim = _compute_viewing_direction_similarity(poses[i], poses[ref_idx])
        
        if dist < min_distance:
            continue
        if dist > max_distance * 3:
            continue
        
        ideal_dist = (min_distance + max_distance) / 2
        dist_score = np.exp(-((dist - ideal_dist) / max_distance) ** 2)
        dir_score = max(0, (dir_sim + 1) / 2)
        scores[i] = 0.6 * dir_score + 0.4 * dist_score
    
    while len(selected) < num_views:
        best_idx = -1
        best_score = -np.inf
        for i in range(n):
            if i in selected:
                continue
            if scores[i] <= -np.inf:
                continue
            too_close = False
            for j in selected:
                if _compute_camera_distance(poses[i], poses[j]) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
            jittered_score = scores[i] + rng.uniform(0, 0.1)
            if jittered_score > best_score:
                best_score = jittered_score
                best_idx = i
        
        if best_idx < 0:
            remaining = [i for i in range(n) if i not in selected]
            if remaining:
                selected.append(rng.choice(remaining))
            else:
                break
        else:
            selected.append(best_idx)
    
    return selected


def _select_views_for_frames(
    frames: List[Dict],
    num_views: int,
    min_distance: float = 0.3,
    max_distance: float = 2.0,
    seed: Optional[int] = None,
    pose_key: str = 'camera_to_world',
) -> List[int]:
    """Select views from a list of frame dicts using overlap-aware selection.
    
    Args:
        frames: List of frame dicts. Each must have a pose under `pose_key`
                or a 'transform_matrix' that can be converted.
        num_views: Target number of views
        min_distance: Minimum camera distance in meters
        max_distance: Maximum camera distance in meters
        seed: Random seed (None = random)
        pose_key: Key in frame dict for the c2w pose matrix
    
    Returns:
        List of selected frame indices (index 0 = reference)
    """
    # Filter out bad frames
    valid_indices = [i for i, f in enumerate(frames) if not f.get('is_bad', False)]
    
    if len(valid_indices) == 0:
        print("Warning: No valid frames found!")
        return []
    
    # Get poses for valid frames
    valid_poses = []
    for i in valid_indices:
        f = frames[i]
        if pose_key in f:
            valid_poses.append(np.array(f[pose_key], dtype=np.float32))
        elif 'transform_matrix' in f:
            # NerfStudio OpenGL → OpenCV
            c2w_opengl = np.array(f['transform_matrix'], dtype=np.float32)
            convert_mat = np.diag([1, -1, -1, 1]).astype(np.float32)
            valid_poses.append(c2w_opengl @ convert_mat)
        else:
            raise KeyError(f"Frame {i} has no '{pose_key}' or 'transform_matrix'")
    
    selected_local = _select_diverse_views(
        valid_poses, num_views, min_distance, max_distance, seed
    )
    
    selected = [valid_indices[i] for i in selected_local]
    return selected


# ============================================================
# VIEW LOADING
# ============================================================

def load_scene_views(
    scene_dir: str,
    num_views: Optional[int] = None,
    image_pattern: str = "*.jpg",
    min_distance: float = 0.3,
    max_distance: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str]]:
    """
    Load multiple views of a scene from a directory.
    
    Uses OVERLAP-AWARE view selection (same as training) to pick views
    with high visual overlap for meaningful query propagation.
    
    Expected directory structure:
    scene_dir/
        images/           # RGB images
        poses/            # Camera poses (4x4 .npy or .txt files)
        intrinsics.txt    # Camera intrinsics (3x3)
    
    OR ScanNet++ structure:
    scene_dir/
        dslr/
            resized_images/
            nerfstudio/transforms.json
    
    Args:
        scene_dir: Path to scene directory
        num_views: Maximum number of views to load (None = all)
        image_pattern: Glob pattern for images
        min_distance: Minimum camera distance in meters (avoid near-duplicates)
        max_distance: Maximum camera distance in meters (views beyond = low overlap)
        seed: Random seed for view selection (42 = deterministic for inference)
    
    Returns:
        images: [N, 3, H, W] tensor
        camera_poses: [N, 4, 4] tensor
        camera_intrinsics: [N, 3, 3] tensor
        depths: [N, 1, H, W] tensor or None (if not found)
        image_names: list of image stems (e.g. ['DSC01502', 'DSC01891', ...])
    """
    scene_path = Path(scene_dir)
    
    # Try ScanNet++ structure first
    nerfstudio_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    if nerfstudio_path.exists():
        return _load_scannetpp_views(scene_path, num_views, min_distance, max_distance, seed)
    
    # Try ScanNet++ COLMAP structure (fallback if nerfstudio transform missing)
    colmap_path = scene_path / "dslr" / "colmap"
    if colmap_path.exists():
        try:
            return _load_scannetpp_colmap_views(scene_path, num_views, min_distance, max_distance, seed)
        except Exception as e:
            print(f"Failed to load COLMAP views: {e}")
    
    # Fallback to generic structure (no poses → uniform sampling)
    return _load_generic_views(scene_path, num_views, image_pattern)


def _load_scannetpp_views(
    scene_path: Path,
    num_views: Optional[int] = None,
    min_distance: float = 0.3,
    max_distance: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str]]:
    """Load views from ScanNet++ structure using transforms_undistorted.json.
    
    Uses overlap-aware view selection (same as training) to pick views
    with high visual overlap for meaningful query propagation.
    
    Returns:
        images, poses, intrinsics, depths, image_names (list of stems)
    """
    import json
    
    transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    with open(transforms_path, 'r') as f:
        data = json.load(f)
        
    intrinsics_dict = {
        'fl_x': data.get('fl_x', 0),
        'fl_y': data.get('fl_y', 0),
        'cx': data.get('cx', 0),
        'cy': data.get('cy', 0),
        'w': data.get('w', 0),
        'h': data.get('h', 0),
    }
    
    frames = data.get('frames', [])
    # Sort by path for determinism
    frames.sort(key=lambda x: x.get('file_path', ''))
    
    if num_views is not None and len(frames) > num_views:
        # OVERLAP-AWARE view selection (same algorithm as training)
        # Selects views with high visual overlap for query propagation.
        # Index 0 of selected = reference view.
        selected_indices = _select_views_for_frames(
            frames, num_views,
            min_distance=min_distance,
            max_distance=max_distance,
            seed=seed,
            pose_key='transform_matrix',  # NerfStudio format
        )
        frames = [frames[i] for i in selected_indices]
        print(f"  Overlap-aware view selection: {len(selected_indices)} views from {len(data.get('frames', []))} total")
         
    # Resize transform for EVALUATION resolution
    # Shortest edge = 480, Max size = 640
    resize_tfm = T.ResizeShortestEdge(short_edge_length=480, max_size=640, sample_style="choice")
    
    images_list = []
    poses_list = []
    intrinsics_list = []
    depths_list = []
    image_names_list = []
    
    for frame in frames:
        # Resolve image path
        rel_path = frame['file_path']
        img_name = Path(rel_path).name
        
        # Look in known locations
        possible_dirs = [
            scene_path / "dslr" / "resized_undistorted_images",
            scene_path / "dslr" / "resized_images",
            scene_path / "dslr" / "images",
            # If rel_path is absolute or relative to transforms.json
            transforms_path.parent / Path(rel_path).parent,
        ]
        
        img_path = None
        for d in possible_dirs:
            if (d / img_name).exists():
                img_path = d / img_name
                break
        
        if img_path is None:
            # Try exact path from json relative to transforms file
            p = transforms_path.parent / rel_path
            if p.exists():
                img_path = p
        
        if img_path is None:
            print(f"Warning: Could not find image {rel_path}, skipping.")
            continue
            
        # Load Image
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to read {img_path}: {e}")
            continue
            
        img_np = np.array(pil_img)
        h_orig, w_orig = img_np.shape[:2]
        
        # Calculate scale from Original (JSON) to Loaded
        # JSON w/h refers to the original resolution usually
        scale_x_load = w_orig / intrinsics_dict['w'] if intrinsics_dict['w'] > 0 else 1.0
        scale_y_load = h_orig / intrinsics_dict['h'] if intrinsics_dict['h'] > 0 else 1.0
        
        # Resize Info
        transform = resize_tfm.get_transform(img_np)
        img_resized = transform.apply_image(img_np)
        
        h_new, w_new = img_resized.shape[:2]
        
        # Load Depth
        depth_m = load_depth_map(img_path, scene_path)
        if depth_m is not None:
             depth_resized = cv2.resize(depth_m, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
             depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0) # [1, H, W]
        else:
             depth_tensor = None
             
        # Tensor - Keep 0-255 range matching training!
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() # [3, H, W]
        
        images_list.append(img_tensor)
        image_names_list.append(Path(img_path).stem)
        if depth_tensor is not None:
            depths_list.append(depth_tensor)
            
        # Pose
        c2w_opengl = np.array(frame['transform_matrix'], dtype=np.float32)
        # Convert OpenGL to OpenCV: +Y up, -Z forward -> +Y down, +Z forward
        convert_mat = np.diag([1, -1, -1, 1]).astype(np.float32)
        c2w_opencv = c2w_opengl @ convert_mat
        poses_list.append(torch.from_numpy(c2w_opencv))
        
        # Intrinsics: Scale = (w_new / w_json)
        scale_x_total = w_new / intrinsics_dict['w'] if intrinsics_dict['w'] > 0 else 1.0
        scale_y_total = h_new / intrinsics_dict['h'] if intrinsics_dict['h'] > 0 else 1.0
        
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = intrinsics_dict['fl_x'] * scale_x_total
        K[1, 1] = intrinsics_dict['fl_y'] * scale_y_total
        K[0, 2] = intrinsics_dict['cx'] * scale_x_total
        K[1, 2] = intrinsics_dict['cy'] * scale_y_total
        
        intrinsics_list.append(torch.from_numpy(K))
        
    if not images_list:
        raise ValueError(f"No valid images found in {scene_path}")

    # Check for consistent shapes (MapAnything requires [B, N, 3, H, W])
    # T.ResizeShortestEdge handles aspect ratio, so if input AR varies, output shape varies.
    # We enforce max crop or just fail if differ?
    # For now assume consistent AR input.
    
    images = torch.stack(images_list)
    poses = torch.stack(poses_list)
    intrinsics = torch.stack(intrinsics_list)
    
    if len(depths_list) == len(images_list):
        depths = torch.stack(depths_list)
    else:
        depths = None
        
    return images, poses, intrinsics, depths, image_names_list


def _load_generic_views(scene_path: Path, num_views: Optional[int] = None, image_pattern: str = "*.jpg") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str]]:
    """Generic loader for images/poses folders.
    
    Returns:
        images, poses, intrinsics, depths, image_names (list of stems)
    """
    img_files = sorted(list(scene_path.glob(image_pattern)))
    if not img_files:
        img_files = sorted(list((scene_path / "images").glob(image_pattern)))
        
    if num_views and len(img_files) > num_views:
        indices = np.linspace(0, len(img_files)-1, num_views, dtype=int)
        img_files = [img_files[i] for i in indices]
        
    # Assume 640x480 default resize
    resize_tfm = T.ResizeShortestEdge(short_edge_length=480, max_size=640, sample_style="choice")
    
    images_list = []
    poses_list = []
    intrinsics_list = []
    
    # Intrinsic (dummy)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = 500
    K[1, 1] = 500
    K[0, 2] = 320
    K[1, 2] = 240
    
    for img_path in img_files:
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        
        transform = resize_tfm.get_transform(img_np)
        img_resized = transform.apply_image(img_np)
        
        images_list.append(torch.from_numpy(img_resized).permute(2, 0, 1).float())
        poses_list.append(torch.eye(4, dtype=torch.float32)) # Dummy pose
        intrinsics_list.append(torch.from_numpy(K))
        
    if not images_list:
        raise ValueError("No images found")
    
    image_names = [f.stem for f in img_files]
    return torch.stack(images_list), torch.stack(poses_list), torch.stack(intrinsics_list), None, image_names


def qvec2rotmat(qvec):
    """
    quaternion to rotmat. qvec = [w, x, y, z]
    """
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y*y - 2 * z*z,     2 * x*y - 2 * z*w,     2 * x*z + 2 * y*w],
        [    2 * x*y + 2 * z*w, 1 - 2 * x*x - 2 * z*z,     2 * y*z - 2 * x*w],
        [    2 * x*z - 2 * y*w,     2 * y*z + 2 * x*w, 1 - 2 * x*x - 2 * y*y]
    ])


def _load_scannetpp_colmap_views(
    scene_path: Path,
    num_views: Optional[int] = None,
    min_distance: float = 0.3,
    max_distance: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str]]:
    """
    Load views from ScanNet++ COLMAP structure.
    Uses overlap-aware view selection (same as training).
    
    Returns:
        images, poses, intrinsics, depths, image_names (list of stems)
    """
    print(f"Loading views from COLMAP: {scene_path}/dslr/colmap")
    
    colmap_dir = scene_path / "dslr" / "colmap"
    images_txt = colmap_dir / "images.txt"
    cameras_txt = colmap_dir / "cameras.txt"
    
    if not images_txt.exists() or not cameras_txt.exists():
         raise FileNotFoundError(f"COLMAP files not found in {colmap_dir}")

    # 1. Read Cameras
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            # params vary by model. PINHOLE: fx, fy, cx, cy
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}

    # 2. Read Images
    images_data = []
    with open(images_txt, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        images_data.append({
            'image_id': image_id, 'qvec': qvec, 'tvec': tvec, 
            'camera_id': camera_id, 'name': name
        })
        i += 2 # Skip points line

    # Sort by name
    images_data.sort(key=lambda x: x['name'])
    
    # Overlap-aware view selection (same as training)
    if num_views and len(images_data) > num_views:
        # Build c2w poses for overlap scoring
        colmap_poses = []
        for img_d in images_data:
            R = qvec2rotmat(img_d['qvec'])
            t = img_d['tvec']
            C2W = np.eye(4, dtype=np.float32)
            C2W[:3, :3] = R.T
            C2W[:3, 3] = -R.T @ t
            colmap_poses.append(C2W)
        selected_indices = _select_diverse_views(
            colmap_poses, num_views,
            min_distance=min_distance,
            max_distance=max_distance,
            seed=seed,
        )
        images_data = [images_data[j] for j in selected_indices]
        print(f"  Overlap-aware view selection: {len(selected_indices)} views from COLMAP")

    # Resize transform (Match Training/Evaluation)
    resize_tfm = T.ResizeShortestEdge(short_edge_length=480, max_size=640, sample_style="choice")
    
    images_list = []
    poses_list = []
    intrinsics_list = []
    depths_list = []
    image_names_list = []
    
    for img_data in images_data:
        img_name = img_data['name']
        
        # Determine image path - Try Undistorted first
        possible_paths = [
            scene_path / "dslr" / "resized_undistorted_images" / img_name,
            scene_path / "dslr" / "resized_images" / img_name,
            scene_path / "dslr" / "images" / img_name,
        ]
        
        img_path = None
        for p in possible_paths:
            if p.exists():
                img_path = p
                break
                
        if img_path is None:
            # Try recursive search if name contains folder
            if "/" in img_name:
                 # COLMAP often just has name.jpg, but maybe dataset has subfolders?
                 pass 
            print(f"Warning: Could not find image {img_name}, skipping.")
            continue
            
        # Load Image
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except:
             continue
        
        img_np = np.array(pil_img)
        h_loaded, w_loaded = img_np.shape[:2]
        
        # Get Camera Params
        cam = cameras[img_data['camera_id']]
        h_orig, w_orig = cam['height'], cam['width']
        
        # Get Intrinsics from COLMAP (Approximation for distorted/undistorted match)
        params = cam['params']
        if cam['model'] in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "SIMPLE_RADIAL", "RADIAL"]:
             # Approx: first param is f (or fx), second is fy (or cx in SIMPLE)
             if cam['model'] == "SIMPLE_RADIAL":
                 fx = fy = params[0]
                 cx, cy = params[1], params[2]
             elif cam['model'] == "RADIAL":
                 fx = fy = params[0]
                 cx, cy = params[1], params[2]
             else: # PINHOLE, OPENCV, OPENCV_FISHEYE
                 fx, fy = params[0], params[1]
                 cx, cy = params[2], params[3]
        else:
             # Fallback
             fx = fy = params[0]
             cx, cy = w_orig/2, h_orig/2
             
        # Scale intrinsics from Original (COLMAP) to Loaded Image
        # Note: If we load "resized_images", dimensions changed.
        # If we load "images" (original), dimensions should match w_orig
        scale_x_load = w_loaded / w_orig
        scale_y_load = h_loaded / h_orig
        
        # Update intrinsics to Loaded resolution
        fx *= scale_x_load
        fy *= scale_y_load
        cx *= scale_x_load
        cy *= scale_y_load

        # Apply Inference Resize
        transform = resize_tfm.get_transform(img_np)
        img_resized = transform.apply_image(img_np)
        h_new, w_new = img_resized.shape[:2]

        # Load Depth
        depth_m = load_depth_map(img_path, scene_path)
        if depth_m is not None:
             depth_resized = cv2.resize(depth_m, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
             depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)
        else:
             depth_tensor = None

        # Tensors
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        
        images_list.append(img_tensor)
        image_names_list.append(Path(img_name).stem)
        if depth_tensor is not None:
             depths_list.append(depth_tensor)

        # Pose: W2C (qvec, tvec) -> C2W
        R = qvec2rotmat(img_data['qvec'])
        t = img_data['tvec']
        
        # C2W = [R^T, -R^T*t]
        R_c2w = R.T
        t_c2w = -R.T @ t
        
        C2W = np.eye(4)
        C2W[:3, :3] = R_c2w
        C2W[:3, 3] = t_c2w
        
        poses_list.append(torch.from_numpy(C2W.astype(np.float32)))
        
        # Intrinsics for Resized Image
        scale_x = w_new / w_loaded
        scale_y = h_new / h_loaded
        
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx * scale_x
        K[1, 1] = fy * scale_y
        K[0, 2] = cx * scale_x
        K[1, 2] = cy * scale_y
        intrinsics_list.append(torch.from_numpy(K))

    if not images_list:
         raise ValueError(f"No valid images found in {scene_path} structure")

    images = torch.stack(images_list)
    poses = torch.stack(poses_list)
    intrinsics = torch.stack(intrinsics_list)
    depths = torch.stack(depths_list) if len(depths_list) == len(images_list) else None
    
    return images, poses, intrinsics, depths, image_names_list


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

def load_ply_pointcloud(ply_path: str) -> PanopticPointCloud:
    """Load a PanopticPointCloud from a PLY file saved by save_ply()."""
    points = []
    colors = []
    instance_ids = []
    semantic_classes = []
    confidences = []
    
    with open(ply_path, 'r') as f:
        # Skip header
        for line in f:
            if line.strip() == "end_header":
                break
        # Read data
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            inst_id = int(parts[6])
            sem_cls = int(parts[7])
            conf = float(parts[8])
            
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
            instance_ids.append(inst_id)
            semantic_classes.append(sem_cls)
            confidences.append(conf)
    
    return PanopticPointCloud(
        points=np.array(points, dtype=np.float32),
        instance_ids=np.array(instance_ids, dtype=np.int32),
        semantic_classes=np.array(semantic_classes, dtype=np.int32),
        confidences=np.array(confidences, dtype=np.float32),
        colors=np.array(colors, dtype=np.float32) if colors else None,
    )


def load_gt_pointcloud_from_views(
    scene_dir: str,
    panoptic_dir: str,
    images: torch.Tensor,       # [N, 3, H, W]
    poses: torch.Tensor,        # [N, 4, 4]
    intrinsics: torch.Tensor,   # [N, 3, 3]
    depths: Optional[torch.Tensor] = None,  # [N, H, W]
    image_names: Optional[List[str]] = None,
) -> PanopticPointCloud:
    """
    Build a GT panoptic point cloud from the same views used for inference.
    
    Reads the panoptic PNG + JSON for each view, lifts GT labels to 3D
    using depth + camera params, and fuses into a single point cloud.
    
    Args:
        scene_dir: Path to scene (e.g. .../data/2e74812d00)
        panoptic_dir: Path to panoptic annotations dir (e.g. .../panoptic)
        images: [N, 3, H, W] input images (for RGB coloring)
        poses: [N, 4, 4] camera-to-world transforms
        intrinsics: [N, 3, 3] camera intrinsics
        depths: [N, H, W] depth maps in meters
        image_names: List of image stems (e.g. ['DSC07231', ...])
    """
    import json
    
    scene_id = Path(scene_dir).name
    panoptic_scene_dir = Path(panoptic_dir) / scene_id
    
    if not panoptic_scene_dir.exists():
        print(f"  [GT] No panoptic annotations found at {panoptic_scene_dir}")
        return None
    
    N = images.shape[0]
    H, W = images.shape[2], images.shape[3]
    
    all_points = []
    all_instance_ids = []
    all_classes = []
    all_colors = []
    
    # Load semantic class names for display
    metadata_path = Path(scene_dir).parent.parent / "metadata" / "semantic_classes.txt"
    class_names = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            for idx, line in enumerate(f):
                class_names[idx] = line.strip()
    
    LABEL_DIVISOR = 10000  # Must match convert_to_panoptic_format.py
    
    for view_idx in range(N):
        if image_names is None or view_idx >= len(image_names):
            # Try to find any annotation file
            json_files = sorted(panoptic_scene_dir.glob("*.json"))
            if view_idx >= len(json_files):
                continue
            stem = json_files[view_idx].stem
        else:
            stem = image_names[view_idx]
        
        json_path = panoptic_scene_dir / f"{stem}.json"
        png_path = panoptic_scene_dir / f"{stem}.png"
        
        if not json_path.exists() or not png_path.exists():
            print(f"  [GT] Missing annotation for view {view_idx} ({stem})")
            continue
        
        # Load panoptic PNG (BGR encoding: panoptic_id = B + G*256 + R*65536)
        pan_img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if pan_img is None:
            continue
        
        # Resize to match inference resolution
        pan_h, pan_w = pan_img.shape[:2]
        if (pan_h, pan_w) != (H, W):
            pan_img = cv2.resize(pan_img, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Decode panoptic IDs (BGR format from cv2.imread)
        pan_ids = (pan_img[:, :, 0].astype(np.int64) + 
                   pan_img[:, :, 1].astype(np.int64) * 256 + 
                   pan_img[:, :, 2].astype(np.int64) * 65536)
        
        # Load segments info
        with open(json_path, 'r') as f:
            segments_info = json.load(f)
        
        # Build ID → category mapping
        id_to_cat = {}
        for seg in segments_info:
            id_to_cat[seg['id']] = seg['category_id']
        
        # Get depth for this view
        if depths is not None and view_idx < depths.shape[0]:
            depth = depths[view_idx]
            if depth.dim() == 3:
                depth = depth.squeeze(0)
            depth_np = depth.cpu().numpy()
        else:
            continue  # Can't lift without depth
        
        # Get pose and intrinsics
        pose = poses[view_idx].cpu().numpy()  # [4, 4]
        K = intrinsics[view_idx].cpu().numpy()  # [3, 3]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create pixel grid
        v_coords, u_coords = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # Valid mask: has depth and is not background (pan_id > 0)
        valid = (depth_np > 0) & (pan_ids > 0)
        
        if valid.sum() == 0:
            continue
        
        # Unproject to 3D
        x_cam = (u_coords - cx) * depth_np / fx
        y_cam = (v_coords - cy) * depth_np / fy
        z_cam = depth_np
        
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)  # [H, W, 4]
        pts_world = pts_cam @ pose.T  # [H, W, 4]
        pts_world = pts_world[..., :3]
        
        # Extract valid points
        pts = pts_world[valid]  # [P, 3]
        pan_ids_valid = pan_ids[valid]
        
        # Decode semantic class from panoptic ID
        sem_classes = (pan_ids_valid // LABEL_DIVISOR).astype(np.int32)
        inst_ids = pan_ids_valid.astype(np.int32)
        
        # Get RGB colors from image
        img_np = images[view_idx].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        rgb = img_np[valid]
        
        all_points.append(pts)
        all_instance_ids.append(inst_ids)
        all_classes.append(sem_classes)
        all_colors.append(rgb)
        
        print(f"  [GT] View {view_idx} ({stem}): {len(pts)} points, "
              f"{len(np.unique(sem_classes))} classes, {len(segments_info)} segments")
    
    if not all_points:
        print("  [GT] No valid GT points generated")
        return None
    
    points = np.concatenate(all_points, axis=0)
    instance_ids = np.concatenate(all_instance_ids, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    print(f"  [GT] Total: {len(points)} points, {len(np.unique(classes))} unique classes")
    
    return PanopticPointCloud(
        points=points,
        instance_ids=instance_ids,
        semantic_classes=classes,
        confidences=np.ones(len(points), dtype=np.float32),
        colors=colors,
    )


def _generate_class_colormap(num_classes: int, seed: int = 42) -> np.ndarray:
    """Generate a deterministic, visually distinct colormap for semantic classes."""
    np.random.seed(seed)
    # Use a mix of tab20 and random for large class counts
    if num_classes <= 20:
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes, :3]
    else:
        # First 20 from tab20, rest random but distinct
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
        extra = np.random.rand(num_classes - 20, 3)
        # Ensure brightness
        extra = 0.3 + 0.6 * extra
        cmap = np.concatenate([tab20, extra], axis=0)
    return cmap


def visualize_panoptic_pointcloud(
    pcd: PanopticPointCloud,
    output_path: str,
    title: str = "Panoptic Point Cloud",
    class_names: Optional[Dict[int, str]] = None,
    max_points: int = 50000,
    point_size: float = 1.0,
    views: Optional[List[Tuple[float, float]]] = None,
):
    """
    Render a PanopticPointCloud as matplotlib 3D scatter plots from multiple angles.
    
    Generates 3 panels:
    1. RGB-colored point cloud (original image colors)
    2. Semantic-colored point cloud (class-based coloring)
    3. Instance-colored point cloud (per-instance coloring)
    
    Each panel shows the same viewpoint. Multiple viewpoints are rendered as separate rows.
    
    Args:
        pcd: PanopticPointCloud to visualize
        output_path: Path to save the PNG
        title: Figure title
        class_names: Optional dict mapping class_id -> name
        max_points: Downsample to this many points for rendering speed
        point_size: Scatter point size
        views: List of (elevation, azimuth) tuples for camera angles
    """
    if len(pcd.points) == 0:
        print(f"  [VIZ] Empty point cloud, skipping visualization")
        return
    
    if views is None:
        views = [(30, -60), (30, 60), (90, 0)]  # Front-ish, back-ish, top-down
    
    # Downsample if needed
    P = len(pcd.points)
    if P > max_points:
        idx = np.random.choice(P, max_points, replace=False)
        pts = pcd.points[idx]
        rgb_colors = pcd.colors[idx] if pcd.colors is not None else None
        sem_classes = pcd.semantic_classes[idx]
        inst_ids = pcd.instance_ids[idx]
    else:
        pts = pcd.points
        rgb_colors = pcd.colors
        sem_classes = pcd.semantic_classes
        inst_ids = pcd.instance_ids
    
    # Generate semantic colormap
    unique_classes = np.unique(sem_classes)
    max_cls = max(unique_classes.max() + 1, 1) if len(unique_classes) > 0 else 1
    class_cmap = _generate_class_colormap(int(max_cls))
    sem_colors = class_cmap[np.clip(sem_classes, 0, len(class_cmap) - 1)]
    
    # Generate instance colormap (random per-instance color)
    unique_insts = np.unique(inst_ids)
    inst_color_map = {}
    np.random.seed(123)
    for inst in unique_insts:
        inst_color_map[inst] = np.random.rand(3) * 0.7 + 0.15  # avoid too dark/bright
    inst_colors = np.array([inst_color_map[i] for i in inst_ids])
    
    # Default RGB colors if not available
    if rgb_colors is None:
        rgb_colors = np.full((len(pts), 3), 0.5)
    
    n_views = len(views)
    fig = plt.figure(figsize=(18, 6 * n_views))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    color_sets = [
        (rgb_colors, "RGB Colors"),
        (sem_colors, "Semantic Classes"),
        (inst_colors, "Instance IDs"),
    ]
    
    for row, (elev, azim) in enumerate(views):
        for col, (colors, label) in enumerate(color_sets):
            ax = fig.add_subplot(n_views, 3, row * 3 + col + 1, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                      c=colors, s=point_size, alpha=0.7, edgecolors='none')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
            # Set equal aspect ratio
            max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
            mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Add class legend if we have class names
    if class_names and len(unique_classes) <= 50:
        legend_lines = []
        for cls_id in sorted(unique_classes):
            name = class_names.get(int(cls_id), f"class_{cls_id}")
            color = class_cmap[min(int(cls_id), len(class_cmap) - 1)]
            legend_lines.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, markersize=8,
                                           label=f"{cls_id}: {name}"))
        fig.legend(handles=legend_lines, loc='lower center', ncol=min(6, len(legend_lines)),
                  fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.01))
        plt.subplots_adjust(bottom=0.08 + 0.02 * (len(legend_lines) // 6))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [VIZ] Saved point cloud visualization to {output_path}")


def visualize_prediction_vs_gt(
    pred_pcd: PanopticPointCloud,
    gt_pcd: PanopticPointCloud,
    output_path: str,
    class_names: Optional[Dict[int, str]] = None,
    max_points: int = 50000,
    point_size: float = 1.0,
):
    """
    Side-by-side comparison of predicted vs ground truth panoptic point clouds.
    
    Layout: 2 rows x 3 columns
      Row 1: Prediction (RGB, Semantic, Instance)
      Row 2: Ground Truth (RGB, Semantic, Instance)
    
    Uses the same viewpoint and color scheme for fair comparison.
    """
    if len(pred_pcd.points) == 0 and (gt_pcd is None or len(gt_pcd.points) == 0):
        print("  [VIZ] Both point clouds empty, skipping comparison")
        return
    
    # Find a good viewpoint using the larger point cloud
    ref_pcd = gt_pcd if (gt_pcd is not None and len(gt_pcd.points) > len(pred_pcd.points)) else pred_pcd
    
    # Use consistent elevation/azimuth
    elev, azim = 30, -60
    
    # Shared class colormap
    all_classes = set(pred_pcd.semantic_classes.tolist())
    if gt_pcd is not None:
        all_classes |= set(gt_pcd.semantic_classes.tolist())
    max_cls = max(max(all_classes) + 1, 1) if all_classes else 1
    class_cmap = _generate_class_colormap(int(max_cls))
    
    def _prepare(pcd, max_pts):
        P = len(pcd.points)
        if P > max_pts:
            idx = np.random.choice(P, max_pts, replace=False)
        else:
            idx = np.arange(P)
        pts = pcd.points[idx]
        rgb = pcd.colors[idx] if pcd.colors is not None else np.full((len(idx), 3), 0.5)
        sem = pcd.semantic_classes[idx]
        inst = pcd.instance_ids[idx]
        
        sem_colors = class_cmap[np.clip(sem, 0, len(class_cmap) - 1)]
        
        unique_inst = np.unique(inst)
        np.random.seed(123)
        imap = {}
        for i in unique_inst:
            imap[i] = np.random.rand(3) * 0.7 + 0.15
        inst_colors = np.array([imap[i] for i in inst])
        
        return pts, rgb, sem_colors, inst_colors, sem, inst
    
    pred_data = _prepare(pred_pcd, max_points)
    gt_data = _prepare(gt_pcd, max_points) if gt_pcd is not None else None
    
    # Compute shared axis limits from both point clouds
    all_pts = [pred_data[0]]
    if gt_data is not None:
        all_pts.append(gt_data[0])
    all_pts_cat = np.concatenate(all_pts, axis=0)
    max_range = (all_pts_cat.max(axis=0) - all_pts_cat.min(axis=0)).max() / 2
    mid = (all_pts_cat.max(axis=0) + all_pts_cat.min(axis=0)) / 2
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Prediction vs Ground Truth", fontsize=18, fontweight='bold', y=0.98)
    
    rows = [("Prediction", pred_data), ("Ground Truth", gt_data)]
    col_labels = ["RGB Colors", "Semantic Classes", "Instance IDs"]
    
    for row_idx, (row_label, data) in enumerate(rows):
        if data is None:
            # Draw "No GT available" text
            for col_idx in range(3):
                ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1, projection='3d')
                ax.text2D(0.5, 0.5, f"No {row_label}\navailable", 
                         transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f"{row_label}: {col_labels[col_idx]}", fontsize=11)
            continue
        
        pts, rgb, sem_c, inst_c, _, _ = data
        color_list = [rgb, sem_c, inst_c]
        
        for col_idx, (colors, col_label) in enumerate(zip(color_list, col_labels)):
            ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=colors, s=point_size, alpha=0.7, edgecolors='none')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_title(f"{row_label}: {col_label}", fontsize=11, fontweight='bold')
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Add class legend
    if class_names:
        pred_classes = set(pred_data[4].tolist())
        gt_classes = set(gt_data[4].tolist()) if gt_data is not None else set()
        shown_classes = sorted(pred_classes | gt_classes)
        if len(shown_classes) <= 30:
            legend_lines = []
            for cls_id in shown_classes:
                name = class_names.get(int(cls_id), f"class_{cls_id}")
                color = class_cmap[min(int(cls_id), len(class_cmap) - 1)]
                legend_lines.append(plt.Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor=color, markersize=8,
                                               label=f"{cls_id}: {name}"))
            fig.legend(handles=legend_lines, loc='lower center',
                      ncol=min(6, len(legend_lines)),
                      fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.01))
    
    # Add stats text
    pred_stats = (f"Pred: {len(pred_pcd.points)} pts, "
                  f"{len(np.unique(pred_pcd.semantic_classes))} classes, "
                  f"{len(np.unique(pred_pcd.instance_ids))} instances")
    gt_stats = ""
    if gt_pcd is not None:
        gt_stats = (f"GT: {len(gt_pcd.points)} pts, "
                    f"{len(np.unique(gt_pcd.semantic_classes))} classes, "
                    f"{len(np.unique(gt_pcd.instance_ids))} instances")
    fig.text(0.02, 0.01, f"{pred_stats}  |  {gt_stats}", fontsize=8, 
             style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [VIZ] Saved pred vs GT comparison to {output_path}")


def visualize_2d_panoptic_views(
    images: torch.Tensor,                        # [N, 3, H, W]
    per_view_preds: List[Dict[str, torch.Tensor]],  # per-view mask predictions
    output_path: str,
    panoptic_dir: Optional[str] = None,
    scene_dir: Optional[str] = None,
    image_names: Optional[List[str]] = None,
    class_names: Optional[Dict[int, str]] = None,
):
    """
    Render per-view 2D panoptic predictions directly from mask logits,
    and optionally load GT panoptic PNGs for comparison.
    
    This is MUCH better than projecting 3D→2D because:
    - No information loss from voxelization
    - Full pixel coverage (every pixel gets a label)
    - GT is loaded directly from annotation PNGs (pixel-perfect)
    
    Layout: N rows × 5 columns (with GT) or 3 columns (without GT)
      Col 0: Original RGB image
      Col 1: Predicted semantic overlay (from 2D masks)
      Col 2: Predicted instance overlay (from 2D masks)
      Col 3: GT semantic overlay (from panoptic PNGs, if available)
      Col 4: GT instance overlay (from panoptic PNGs, if available)
    """
    import json as json_module
    
    N, C, H, W = images.shape
    
    # Determine max class ID for consistent colormap
    max_cls = 1
    for pred in per_view_preds:
        if len(pred['pred_classes']) > 0:
            max_cls = max(max_cls, int(pred['pred_classes'].max().item()) + 1)
    
    # Check if GT is available
    has_gt = (panoptic_dir is not None and scene_dir is not None 
              and image_names is not None)
    gt_panoptic_maps = []
    LABEL_DIVISOR = 10000
    
    if has_gt:
        scene_id = Path(scene_dir).name
        panoptic_scene_dir = Path(panoptic_dir) / scene_id
        
        for view_idx in range(N):
            stem = image_names[view_idx] if view_idx < len(image_names) else None
            if stem is None:
                gt_panoptic_maps.append(None)
                continue
                
            json_path = panoptic_scene_dir / f"{stem}.json"
            png_path = panoptic_scene_dir / f"{stem}.png"
            
            if not json_path.exists() or not png_path.exists():
                gt_panoptic_maps.append(None)
                continue
            
            # Load panoptic PNG (BGR encoding: panoptic_id = B + G*256 + R*65536)
            pan_img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if pan_img is None:
                gt_panoptic_maps.append(None)
                continue
                
            # Resize to match inference resolution
            pan_img = cv2.resize(pan_img, (W, H), interpolation=cv2.INTER_NEAREST)
            
            b, g, r = pan_img[:,:,0].astype(np.int64), pan_img[:,:,1].astype(np.int64), pan_img[:,:,2].astype(np.int64)
            panoptic_ids = b + g * 256 + r * 65536
            semantic_ids = panoptic_ids // LABEL_DIVISOR
            
            # Load segments_info for class mapping
            with open(json_path, 'r') as f:
                seg_info = json_module.load(f)
            
            # Build panoptic_id → category_id mapping
            # JSON is a list of segment dicts directly (not wrapped in 'segments_info')
            id_to_cat = {}
            segments_list = seg_info if isinstance(seg_info, list) else seg_info.get('segments_info', [])
            for seg in segments_list:
                id_to_cat[seg['id']] = seg['category_id']
            
            max_cls = max(max_cls, int(semantic_ids.max()) + 1)
            gt_panoptic_maps.append({
                'semantic': semantic_ids,
                'panoptic_ids': panoptic_ids,
            })
    
    if not has_gt or all(g is None for g in gt_panoptic_maps):
        has_gt = False
    
    class_cmap = _generate_class_colormap(int(max_cls))
    
    n_cols = 5 if has_gt else 3
    fig, axes = plt.subplots(N, n_cols, figsize=(5 * n_cols, 4 * N))
    if N == 1:
        axes = axes[np.newaxis, :]
    
    fig.suptitle("Per-View Panoptic Predictions" + (" + GT" if has_gt else ""),
                 fontsize=16, fontweight='bold')
    
    for view_idx in range(N):
        # Get image as [H, W, 3] in [0, 1]
        img = images[view_idx].permute(1, 2, 0).cpu().numpy()
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        
        # --- Build 2D semantic and instance maps from per-view predictions ---
        pred = per_view_preds[view_idx]
        pred_masks = pred['pred_masks'].cpu()       # [Q, H', W'] probabilities
        pred_classes = pred['pred_classes'].cpu()    # [Q]
        pred_scores = pred['pred_scores'].cpu()      # [Q]
        Q = pred_masks.shape[0]
        
        # Resize masks to image resolution if needed
        if pred_masks.shape[-2:] != (H, W):
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )[0]
        
        # Assign each pixel to its best query
        weighted_masks = pred_masks * pred_scores.view(Q, 1, 1)
        instance_map = weighted_masks.argmax(dim=0).numpy()   # [H, W] query idx
        semantic_map = pred_classes[instance_map].numpy()      # [H, W] class id
        max_score_map = weighted_masks.max(dim=0)[0].numpy()   # [H, W]
        
        # Build colored overlays
        sem_colors = class_cmap[np.clip(semantic_map, 0, len(class_cmap) - 1)]  # [H, W, 3]
        
        # Instance colormap
        unique_inst = np.unique(instance_map)
        np.random.seed(123)
        inst_cmap = {}
        for i in unique_inst:
            inst_cmap[i] = np.random.rand(3) * 0.7 + 0.15
        inst_colors = np.zeros((H, W, 3), dtype=np.float64)
        for i in unique_inst:
            mask = instance_map == i
            inst_colors[mask] = inst_cmap[i]
        
        # Create alpha mask: stronger overlay where prediction is more confident
        alpha = np.clip(max_score_map * 2.0, 0.3, 0.7)  # at least 0.3 overlay
        alpha_3d = alpha[:, :, np.newaxis]
        
        # Col 0: RGB
        view_name = image_names[view_idx] if image_names and view_idx < len(image_names) else f"View {view_idx}"
        axes[view_idx, 0].imshow(img)
        axes[view_idx, 0].set_title(f"{view_name}: RGB", fontsize=10)
        axes[view_idx, 0].axis('off')
        
        # Col 1: Pred semantic (confidence-weighted blend)
        blended_sem = (1 - alpha_3d) * img + alpha_3d * sem_colors
        axes[view_idx, 1].imshow(np.clip(blended_sem, 0, 1))
        n_unique_cls = len(np.unique(semantic_map))
        axes[view_idx, 1].set_title(f"{view_name}: Pred Semantic ({n_unique_cls} cls)", fontsize=10)
        axes[view_idx, 1].axis('off')
        
        # Add text labels at segment centroids for predicted semantic
        _add_segment_labels(axes[view_idx, 1], semantic_map, pred_classes.numpy(),
                           instance_map, class_names, max_labels=15)
        
        # Col 2: Pred instance
        blended_inst = (1 - alpha_3d) * img + alpha_3d * inst_colors
        axes[view_idx, 2].imshow(np.clip(blended_inst, 0, 1))
        n_unique_inst = len(np.unique(instance_map))
        axes[view_idx, 2].set_title(f"{view_name}: Pred Instance ({n_unique_inst} inst)", fontsize=10)
        axes[view_idx, 2].axis('off')
        
        # Add text labels at segment centroids for predicted instance
        # Show "class_name #N" to distinguish instances of the same class
        _add_segment_labels(axes[view_idx, 2], semantic_map, pred_classes.numpy(),
                           instance_map, class_names, max_labels=15,
                           show_instance_id=True)
        
        # Col 3: GT semantic (from panoptic PNGs)
        if has_gt:
            gt_data = gt_panoptic_maps[view_idx] if view_idx < len(gt_panoptic_maps) else None
            if gt_data is not None:
                gt_sem = gt_data['semantic']
                gt_sem_colors = class_cmap[np.clip(gt_sem, 0, len(class_cmap) - 1)]
                blended_gt = 0.4 * img + 0.6 * gt_sem_colors
                n_gt_cls = len(np.unique(gt_sem))
                axes[view_idx, 3].imshow(np.clip(blended_gt, 0, 1))
                axes[view_idx, 3].set_title(f"{view_name}: GT Semantic ({n_gt_cls} cls)", fontsize=10)
                
                # Add text labels for GT segments
                gt_inst_map = gt_data['panoptic_ids']  # unique per segment
                _add_segment_labels(axes[view_idx, 3], gt_sem, None,
                                   gt_inst_map, class_names, max_labels=20)
                
                # Col 4: GT instance (from panoptic PNGs)
                gt_unique_inst = np.unique(gt_inst_map)
                np.random.seed(456)  # Different seed from pred instances
                gt_inst_cmap = {}
                for inst_id in gt_unique_inst:
                    gt_inst_cmap[inst_id] = np.random.rand(3) * 0.7 + 0.15
                gt_inst_colors = np.zeros((H, W, 3), dtype=np.float64)
                for inst_id in gt_unique_inst:
                    mask = gt_inst_map == inst_id
                    gt_inst_colors[mask] = gt_inst_cmap[inst_id]
                blended_gt_inst = 0.4 * img + 0.6 * gt_inst_colors
                n_gt_inst = len(gt_unique_inst)
                axes[view_idx, 4].imshow(np.clip(blended_gt_inst, 0, 1))
                axes[view_idx, 4].set_title(f"{view_name}: GT Instance ({n_gt_inst} inst)", fontsize=10)
                
                _add_segment_labels(axes[view_idx, 4], gt_sem, None,
                                   gt_inst_map, class_names, max_labels=20,
                                   show_instance_id=True)
            else:
                axes[view_idx, 3].imshow(img)
                axes[view_idx, 3].set_title(f"{view_name}: GT (N/A)", fontsize=10)
                axes[view_idx, 4].imshow(img)
                axes[view_idx, 4].set_title(f"{view_name}: GT (N/A)", fontsize=10)
            axes[view_idx, 3].axis('off')
            axes[view_idx, 4].axis('off')
    
    # Add class legend at the bottom
    all_pred_classes = set()
    all_gt_classes = set()
    for pred in per_view_preds:
        all_pred_classes.update(pred['pred_classes'].cpu().numpy().tolist())
    if has_gt:
        for gt_data in gt_panoptic_maps:
            if gt_data is not None:
                all_gt_classes.update(np.unique(gt_data['semantic']).tolist())
    all_classes_shown = sorted(all_pred_classes | all_gt_classes)
    
    if class_names and len(all_classes_shown) <= 40:
        legend_lines = []
        for cls_id in all_classes_shown:
            name = class_names.get(int(cls_id), f"cls_{cls_id}")
            color = class_cmap[min(int(cls_id), len(class_cmap) - 1)]
            legend_lines.append(plt.Line2D([0], [0], marker='s', color='w',
                                           markerfacecolor=color, markersize=8,
                                           label=f"{cls_id}: {name}"))
        fig.legend(handles=legend_lines, loc='lower center', 
                  ncol=min(6, len(legend_lines)),
                  fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [VIZ] Saved 2D per-view visualization to {output_path}")


def _add_segment_labels(
    ax,
    semantic_map: np.ndarray,    # [H, W] class IDs
    pred_classes_arr: Optional[np.ndarray],  # [Q] query→class mapping (None for GT)
    instance_map: np.ndarray,    # [H, W] instance/query IDs
    class_names: Optional[Dict[int, str]],
    max_labels: int = 15,
    min_segment_pixels: int = 500,
    show_instance_id: bool = False,
):
    """
    Add text labels at the centroid of the largest segments in a panoptic map.
    
    Places short class name text with a semi-transparent background at each
    segment's centroid for readability.
    
    Args:
        show_instance_id: If True, show "class #N" format for instance view
    """
    if class_names is None:
        return
    
    from scipy import ndimage
    
    unique_instances = np.unique(instance_map)
    
    # Compute area per instance and sort by area (largest first)
    instance_info = []
    for inst_id in unique_instances:
        mask = instance_map == inst_id
        area = mask.sum()
        if area < min_segment_pixels:
            continue
        # Get semantic class for this instance
        cls_id = int(semantic_map[mask][0])
        # Get centroid
        cy, cx = ndimage.center_of_mass(mask)
        instance_info.append((area, cls_id, cx, cy))
    
    # Sort by area descending, take top N
    instance_info.sort(key=lambda x: -x[0])
    instance_info = instance_info[:max_labels]
    
    # Track per-class instance count for instance ID labeling
    class_instance_counter = {}
    
    for area, cls_id, cx, cy in instance_info:
        name = class_names.get(cls_id, f"c{cls_id}")
        # Truncate long names
        if len(name) > 12:
            name = name[:11] + "…"
        if show_instance_id:
            idx = class_instance_counter.get(cls_id, 0)
            class_instance_counter[cls_id] = idx + 1
            name = f"{name} #{idx}"
        ax.text(cx, cy, name, fontsize=5, color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                         alpha=0.6, edgecolor='none'))


def _project_pcd_to_view(
    pcd: PanopticPointCloud,
    pose_c2w: np.ndarray,      # [4, 4]
    K: np.ndarray,             # [3, 3]
    H: int, W: int,
    class_cmap: np.ndarray,
    color_by: str = 'semantic',  # 'semantic' or 'instance'
) -> np.ndarray:
    """Project a 3D point cloud onto a 2D view and return an overlay image."""
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    if len(pcd.points) == 0:
        return overlay
    
    # World to camera: W2C = inv(C2W)
    w2c = np.linalg.inv(pose_c2w)
    
    # Transform points to camera frame
    pts_h = np.concatenate([pcd.points, np.ones((len(pcd.points), 1))], axis=1)  # [P, 4]
    pts_cam = (w2c @ pts_h.T).T[:, :3]  # [P, 3]
    
    # Filter points in front of camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    
    if len(pts_cam) == 0:
        return overlay
    
    # Project to image
    pts_2d = (K @ pts_cam.T).T  # [P, 3]
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]  # [P, 2]
    
    u = np.round(pts_2d[:, 0]).astype(int)
    v = np.round(pts_2d[:, 1]).astype(int)
    
    # Filter in-bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    
    if color_by == 'semantic':
        sem = pcd.semantic_classes[valid][in_bounds]
        colors = class_cmap[np.clip(sem, 0, len(class_cmap) - 1)]
    else:  # instance
        inst = pcd.instance_ids[valid][in_bounds]
        unique_inst = np.unique(inst)
        np.random.seed(123)
        imap = {}
        for i in unique_inst:
            imap[i] = np.random.rand(3) * 0.7 + 0.15
        colors = np.array([imap[i] for i in inst])
    
    # Z-buffer: closer points overwrite farther ones
    depths = pts_cam[:, 2][in_bounds]
    order = np.argsort(-depths)  # Far-to-near (near overwrites)
    overlay[v[order], u[order]] = colors[order]
    
    return overlay


def run_multiview_inference(
    model,
    scene_dir: str,
    output_dir: str,
    num_views: int = 8,
    visualize: bool = True,
    panoptic_dir: Optional[str] = None,
    mask_threshold: float = 0.5,
    min_distance: float = 0.3,
    max_distance: float = 2.0,
):
    """Run inference on a scene with optional visualization and GT comparison.
    
    Args:
        model: Trained MultiViewMask2Former model
        scene_dir: Path to scene data directory
        output_dir: Output directory for PLY and PNG files
        num_views: Number of views to use
        visualize: Whether to generate matplotlib visualizations
        panoptic_dir: Path to panoptic GT annotations (for GT comparison)
        mask_threshold: Threshold for binary mask prediction (lower=more points)
        min_distance: Minimum camera distance for overlap-aware view selection
        max_distance: Maximum camera distance for overlap-aware view selection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading scene from {scene_dir}...")
    images, poses, intrinsics, gt_depths, image_names = load_scene_views(
        scene_dir, num_views,
        min_distance=min_distance,
        max_distance=max_distance,
        seed=42,  # Deterministic for inference reproducibility
    )
    print(f"Loaded {len(images)} views with shape {images.shape}")
    print(f"  View image names: {image_names}")
    
    # NOTE: MultiViewPanopticInference.__call__ expects UNBATCHED inputs [N, ...]
    # It adds batch dimension internally.
    
    if gt_depths is not None:
        # Load returns [N, 1, H, W], but we likely want [N, H, W]
        if gt_depths.dim() == 4 and gt_depths.shape[1] == 1:
            gt_depths = gt_depths.squeeze(1)
        print("Loaded ground truth depth maps.")
    
    inference = MultiViewPanopticInference(model, mask_threshold=mask_threshold)
    
    # Run
    print(f"Running inference (mask_threshold={mask_threshold})...")
    # Pass depth to inference
    pcd, per_view_preds = inference(images, poses, intrinsics, depths=gt_depths)
    
    # Save PLY
    scene_name = Path(scene_dir).name
    ply_path = output_path / f"{scene_name}.ply"
    print(f"Saving to {ply_path}...")
    pcd.save_ply(str(ply_path), use_semantic_colors=True)
    
    # ── Visualization ──
    if visualize:
        print("\n=== Generating Visualizations ===")
        
        # Load class names for legends
        metadata_path = Path(scene_dir).parent.parent / "metadata" / "semantic_classes.txt"
        class_names = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for idx, line in enumerate(f):
                    class_names[idx] = line.strip()
            print(f"  Loaded {len(class_names)} class names from {metadata_path}")
        
        # 1. Prediction-only 3D point cloud visualization (multi-angle)
        viz_pred_path = output_path / f"{scene_name}_pred_3d.png"
        visualize_panoptic_pointcloud(
            pcd, str(viz_pred_path),
            title=f"Predicted Panoptic Point Cloud: {scene_name}",
            class_names=class_names,
        )
        
        # 2. Load GT point cloud for comparison (if panoptic_dir provided)
        gt_pcd = None
        if panoptic_dir:
            print(f"\n  Loading GT point cloud from {panoptic_dir}...")
            gt_pcd = load_gt_pointcloud_from_views(
                scene_dir=scene_dir,
                panoptic_dir=panoptic_dir,
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                depths=gt_depths,
                image_names=image_names,
            )
            
            if gt_pcd is not None:
                # Save GT PLY too
                gt_ply_path = output_path / f"{scene_name}_gt.ply"
                gt_pcd.save_ply(str(gt_ply_path))
                
                # GT-only 3D visualization
                viz_gt_path = output_path / f"{scene_name}_gt_3d.png"
                visualize_panoptic_pointcloud(
                    gt_pcd, str(viz_gt_path),
                    title=f"Ground Truth Panoptic Point Cloud: {scene_name}",
                    class_names=class_names,
                )
        
        # 3. Side-by-side pred vs GT comparison (3D)
        viz_compare_path = output_path / f"{scene_name}_pred_vs_gt_3d.png"
        visualize_prediction_vs_gt(
            pcd, gt_pcd, str(viz_compare_path),
            class_names=class_names,
        )
        
        # 4. Per-view 2D overlay visualization (direct 2D masks, not 3D reprojection)
        viz_2d_path = output_path / f"{scene_name}_2d_views.png"
        try:
            visualize_2d_panoptic_views(
                images, per_view_preds,
                str(viz_2d_path),
                panoptic_dir=panoptic_dir,
                scene_dir=scene_dir,
                image_names=image_names,
                class_names=class_names,
            )
        except Exception as e:
            import traceback
            print(f"  [VIZ] ERROR in 2D visualization: {e}")
            traceback.print_exc()
        
        print(f"\n=== Visualization Complete ===")
        print(f"  3D prediction:        {viz_pred_path}")
        if gt_pcd is not None:
            print(f"  3D ground truth:      {output_path / f'{scene_name}_gt_3d.png'}")
        print(f"  3D pred vs GT:        {viz_compare_path}")
        print(f"  2D per-view overlays: {viz_2d_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="", help="Config file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--num-views", type=int, default=8, help="Number of views")
    parser.add_argument("--panoptic-dir", type=str, default=None,
                        help="Path to panoptic GT annotations for comparison "
                             "(e.g. datasets/scannet/scannetpp/panoptic)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Mask confidence threshold (lower=more points, default 0.5)")
    parser.add_argument("--min-distance", type=float, default=0.3,
                        help="Minimum camera distance for view selection (meters, default 0.3)")
    parser.add_argument("--max-distance", type=float, default=2.0,
                        help="Maximum camera distance for view selection (meters, default 2.0)")
    args = parser.parse_args()
    
    model, cfg = load_multiview_model(args.model, args.config)
    
    run_multiview_inference(
        model,
        args.scene,
        args.output,
        num_views=args.num_views,
        visualize=not args.no_viz,
        panoptic_dir=args.panoptic_dir,
        mask_threshold=args.mask_threshold,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )
