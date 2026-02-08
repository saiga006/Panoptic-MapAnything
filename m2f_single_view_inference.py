"""
Inference script for Mask2Former with MapAnything backbone
Performs:
1. 3D reconstruction (depth + point cloud) from MapAnything
2. Panoptic segmentation from trained Mask2Former
"""

import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import argparse

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
sys.path.insert(0, './Mask2Former')
from mask2former import add_maskformer2_config

# Import the custom backbone from training script
# This registers the MapAnythingWithPanopticDPT with Detectron2
from m2f_train_cluster_working import MapAnythingWithPanopticDPT, build_mapanything_panoptic_dpt_backbone


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def setup_cfg(checkpoint_path, confidence_threshold=0.5):
    """Setup config for inference"""
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    
    # Model architecture (must match training config)
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_panoptic_dpt_backbone"
    
    # MapAnything config
    from detectron2.config import CfgNode as CN
    cfg.MODEL.MAPANYTHING = CN()
    cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH = "./pretrained_models/map_anything/test"
    
    # Backbone features
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # Mask2Former head config (match training)
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 133
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    
    # Pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    
    # Mask Former decoder config
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
    
    # Test config
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = confidence_threshold
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    
    # Input config
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.FORMAT = "RGB"
    
    # Dataset config (required for metadata)
    cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic",)
    cfg.DATASETS.TEST = ("coco_2017_val_panoptic",)
    
    # Additional model config
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    
    # Solver config (required for differential LR info, not used in inference)
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.DPT_LR = 1e-5
    
    # Load trained weights
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg


def load_model(checkpoint_path):
    """Load trained Mask2Former model with MapAnything backbone"""
    print("="*80)
    print("LOADING TRAINED MODEL")
    print("="*80)
    
    # Register COCO panoptic dataset to get metadata
    from m2f_train_cluster_working import register_coco_panoptic
    coco_root = "./datasets/coco"
    register_coco_panoptic(coco_root)
    
    # Setup config
    cfg = setup_cfg(checkpoint_path)
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load checkpoint
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    
    return model, cfg


# ============================================================
# MAPANYTHING 3D RECONSTRUCTION
# ============================================================

def run_mapanything_inference(image_path, mapanything_model):
    """Run MapAnything for 3D reconstruction"""
    print("\n" + "="*80)
    print("MAPANYTHING 3D RECONSTRUCTION")
    print("="*80)
    
    # Load image using MapAnything's loader
    if os.path.isdir(image_path):
        views = load_images(image_path)
    else:
        # Single image - create temp directory
        temp_dir = Path(image_path).parent / "temp_inference"
        temp_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy(image_path, temp_dir / Path(image_path).name)
        views = load_images(str(temp_dir))
        shutil.rmtree(temp_dir)
    
    # Move to device
    device = next(mapanything_model.parameters()).device
    for view in views:
        for key, val in view.items():
            if isinstance(val, torch.Tensor):
                view[key] = val.to(device)
    
    # Run inference
    print("Running MapAnything inference...")
    with torch.no_grad():
        predictions = mapanything_model(views)
    
    # Extract results for first view
    pred = predictions[0]
    
    results = {
        'depth': pred['depth_along_ray'].cpu().numpy().squeeze(),
        'metric_scale': pred['metric_scaling_factor'].item() if isinstance(pred['metric_scaling_factor'], torch.Tensor) else pred['metric_scaling_factor'],
        'pts3d': pred['pts3d'].cpu().numpy().squeeze(),
        'confidence': pred['conf'].cpu().numpy().squeeze(),
        'cam_trans': pred['cam_trans'].cpu().numpy(),
        'cam_quats': pred['cam_quats'].cpu().numpy(),
    }
    
    # Compute metric depth
    results['metric_depth'] = results['depth'] * results['metric_scale']
    
    print(f"✓ 3D reconstruction complete")
    print(f"  Depth range: {results['metric_depth'].min():.2f}m - {results['metric_depth'].max():.2f}m")
    print(f"  Metric scale: {results['metric_scale']:.3f}")
    
    return results


def create_point_cloud(pts3d, confidence, image, conf_threshold=0.5, segmentation_mask=None, segments_info=None, metadata=None):
    """
    Create colored point cloud from 3D points
    
    Args:
        pts3d: 3D points array (H, W, 3)
        confidence: Confidence map (H, W)
        image: PIL Image for RGB coloring
        conf_threshold: Confidence threshold for filtering
        segmentation_mask: Optional panoptic segmentation mask (H, W) with panoptic IDs
        segments_info: Optional list of segment metadata dicts with 'id' and 'category_id'
        metadata: Optional COCO metadata for class colors
    
    Returns:
        pcd: Open3D point cloud (RGB colored)
        pcd_semantic: Optional semantic point cloud (class colored) if segmentation provided
    """
    H, W = pts3d.shape[:2]
    
    # Flatten points and confidence
    points = pts3d.reshape(-1, 3)
    confidences = confidence.reshape(-1)
    
    # Filter by confidence
    valid_mask = confidences > conf_threshold
    points_filtered = points[valid_mask]
    
    # Create Open3D point cloud (RGB version)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    
    # Add RGB colors from image
    img_resized = np.array(image.resize((W, H)))
    colors = img_resized.reshape(-1, 3) / 255.0
    colors_filtered = colors[valid_mask]
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    # Remove outliers
    pcd_cleaned, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"✓ RGB point cloud created: {len(pcd_cleaned.points)} points")
    
    # Create semantic point cloud if segmentation is provided
    pcd_semantic = None
    if segmentation_mask is not None and metadata is not None:
        # CRITICAL: pts3d has shape (H_depth, W_depth, 3)
        # segmentation_mask has shape (H_orig, W_orig)
        # We must resize segmentation to match pts3d resolution EXACTLY
        
        print(f"  DEBUG: Input segmentation mask shape: {segmentation_mask.shape}")
        print(f"  DEBUG: Target pts3d shape: {pts3d.shape}")
        
        # Resize segmentation mask to match point cloud resolution
        # Use int32 to preserve panoptic IDs
        from PIL import Image as PILImage
        seg_mask_int32 = segmentation_mask.astype(np.int32)
        
        # Resize to match pts3d dimensions (H, W) - NOT the image dimensions!
        target_height, target_width = H, W  # These are from pts3d.shape[:2]
        seg_mask_resized = np.array(PILImage.fromarray(seg_mask_int32).resize(
            (target_width, target_height),  # PIL uses (width, height)
            PILImage.NEAREST
        ))
        
        print(f"  DEBUG: Resized segmentation to: {seg_mask_resized.shape} (matching pts3d)")
        
        # Flatten and filter by confidence
        seg_labels = seg_mask_resized.reshape(-1)
        seg_labels_filtered = seg_labels[valid_mask]
        
        # If we have segments_info, use it to map panoptic IDs to category IDs
        # This is the correct way to extract semantic classes from panoptic segmentation
        if segments_info is not None:
            print(f"  Using segments_info to map panoptic IDs to semantic classes")
            
            # Build mapping from panoptic_id -> category_id
            id_to_category = {}
            for segment in segments_info:
                id_to_category[segment['id']] = segment['category_id']
            
            print(f"  Found {len(id_to_category)} segments:")
            for seg_id, cat_id in list(id_to_category.items())[:10]:
                print(f"    Panoptic ID {seg_id} -> Category {cat_id}")
            
            # Map each point's panoptic ID to its semantic class
            semantic_labels = np.array([id_to_category.get(pid, 0) for pid in seg_labels_filtered])
            
        else:
            # Fallback: try to decode panoptic IDs manually
            print(f"  No segments_info available - attempting manual decoding")
            max_id = seg_labels_filtered.max()
            if max_id > 200:
                semantic_labels = seg_labels_filtered // 1000
            else:
                semantic_labels = seg_labels_filtered
        
        print(f"  DEBUG: Panoptic ID range: {seg_labels_filtered.min()} - {seg_labels_filtered.max()}")
        print(f"  DEBUG: Semantic class range: {semantic_labels.min()} - {semantic_labels.max()}")
        print(f"  DEBUG: Unique semantic classes: {sorted(np.unique(semantic_labels).tolist())[:20]}")
        
        # Create color map for semantic classes
        num_classes = 133  # COCO panoptic classes
        unique_classes = np.unique(semantic_labels)
        
        # Use semantic class colors
        if hasattr(metadata, 'stuff_colors'):
            # Use COCO colors if available (already in 0-255 range for point clouds)
            class_colors = np.array(metadata.stuff_colors + [[128, 128, 128]] * (num_classes - len(metadata.stuff_colors)))
        else:
            # Generate distinct colors for each class
            np.random.seed(42)
            class_colors = np.random.randint(0, 255, size=(num_classes, 3))
        
        # Map semantic labels to colors - clamp to valid range
        semantic_labels_clamped = np.clip(semantic_labels, 0, num_classes - 1)
        semantic_colors = class_colors[semantic_labels_clamped] / 255.0
        
        print(f"  Using semantic class-based coloring for {len(unique_classes)} classes")
        
        # Create semantic point cloud
        pcd_semantic = o3d.geometry.PointCloud()
        pcd_semantic.points = o3d.utility.Vector3dVector(points_filtered)
        pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
        
        # Remove outliers (use same indices as RGB cloud)
        pcd_semantic = pcd_semantic.select_by_index(inlier_indices)
        
        print(f"✓ Semantic point cloud created: {len(pcd_semantic.points)} points")
        print(f"  Unique classes: {len(np.unique(semantic_labels))}")
    
    return pcd_cleaned, pcd_semantic


def create_point_cloud_comparison(pcd_rgb, pcd_semantic, output_dir, pts3d, confidence, 
                                   seg_mask, segments_info, metadata, conf_threshold=0.5):
    """
    Create comparison visualizations: RGB vs Semantic, and labeled semantic point cloud
    
    Args:
        pcd_rgb: RGB colored point cloud
        pcd_semantic: Semantic colored point cloud
        output_dir: Directory to save outputs
        pts3d: 3D points array (for extracting labels)
        confidence: Confidence map
        seg_mask: Segmentation mask
        segments_info: Segment metadata with category IDs
        metadata: COCO metadata for class names
        conf_threshold: Confidence threshold
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("  Creating side-by-side RGB vs Semantic comparison...")
    
    # Get point arrays
    rgb_points = np.asarray(pcd_rgb.points)
    rgb_colors = np.asarray(pcd_rgb.colors)
    sem_points = np.asarray(pcd_semantic.points)
    sem_colors = np.asarray(pcd_semantic.colors)
    
    # Downsample for visualization
    max_points = 50000
    if len(rgb_points) > max_points:
        step = len(rgb_points) // max_points
        rgb_points_viz = rgb_points[::step]
        rgb_colors_viz = rgb_colors[::step]
        sem_points_viz = sem_points[::step]
        sem_colors_viz = sem_colors[::step]
    else:
        rgb_points_viz = rgb_points
        rgb_colors_viz = rgb_colors
        sem_points_viz = sem_points
        sem_colors_viz = sem_colors
    
    # Create figure with side-by-side comparison
    fig = plt.figure(figsize=(24, 10))
    
    # RGB Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(rgb_points_viz[:, 0], rgb_points_viz[:, 1], rgb_points_viz[:, 2],
                c=rgb_colors_viz, s=1, alpha=0.6)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title('RGB Point Cloud\n(Original Image Colors)', fontsize=14, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    
    # Semantic Point Cloud with Labels
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sem_points_viz[:, 0], sem_points_viz[:, 1], sem_points_viz[:, 2],
                c=sem_colors_viz, s=1, alpha=0.6)
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_zlabel('Z (m)', fontsize=12)
    ax2.set_title('Semantic Segmentation Point Cloud\n(Colored by Predicted Class)', 
                  fontsize=14, fontweight='bold')
    ax2.view_init(elev=30, azim=45)
    
    # Add class legend to semantic plot
    if segments_info is not None and metadata is not None:
        # Get unique categories
        unique_categories = set()
        for seg in segments_info:
            unique_categories.add(seg['category_id'])
        
        # Create legend with class names and colors
        legend_elements = []
        class_colors_used = set()
        
        for cat_id in sorted(unique_categories):
            if cat_id < len(metadata.stuff_classes):
                class_name = metadata.stuff_classes[cat_id]
            elif hasattr(metadata, 'thing_classes') and (cat_id - len(metadata.stuff_classes)) < len(metadata.thing_classes):
                class_name = metadata.thing_classes[cat_id - len(metadata.stuff_classes)]
            else:
                class_name = f"Class {cat_id}"
            
            # Get color for this class
            if hasattr(metadata, 'stuff_colors') and cat_id < len(metadata.stuff_colors):
                color = np.array(metadata.stuff_colors[cat_id]) / 255.0
            else:
                color = plt.cm.tab20(cat_id % 20)[:3]
            
            color_tuple = tuple(color)
            if color_tuple not in class_colors_used:
                class_colors_used.add(color_tuple)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=color, markersize=10,
                                                  label=class_name))
        
        # Add legend outside the plot
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
                  fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "pointcloud_rgb_vs_semantic.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ RGB vs Semantic comparison saved: {comparison_path}")
    plt.close()
    
    # Create overlay/blend visualization
    print("  Creating blended overlay visualization...")
    create_blended_point_cloud_overlay(pcd_rgb, pcd_semantic, output_dir, segments_info, metadata)


def create_blended_point_cloud_overlay(pcd_rgb, pcd_semantic, output_dir, segments_info=None, metadata=None, alpha=0.5):
    """
    Create a blended visualization overlaying semantic colors on RGB point cloud
    
    Args:
        pcd_rgb: RGB point cloud
        pcd_semantic: Semantic point cloud
        output_dir: Output directory
        segments_info: Segment metadata with category IDs (for labels)
        metadata: COCO metadata for class names
        alpha: Blend factor (0=full RGB, 1=full semantic)
    """
    # Create blended point cloud
    pcd_blend = o3d.geometry.PointCloud()
    pcd_blend.points = pcd_rgb.points
    
    # Blend colors: (1-alpha)*RGB + alpha*Semantic
    rgb_colors = np.asarray(pcd_rgb.colors)
    sem_colors = np.asarray(pcd_semantic.colors)
    blended_colors = (1 - alpha) * rgb_colors + alpha * sem_colors
    pcd_blend.colors = o3d.utility.Vector3dVector(blended_colors)
    
    # Save blended point cloud
    blend_path = os.path.join(output_dir, "reconstruction_blended.ply")
    o3d.io.write_point_cloud(blend_path, pcd_blend)
    print(f"  ✓ Blended point cloud saved: {blend_path}")
    print(f"    (RGB + Semantic overlay with alpha={alpha})")
    
    # Create matplotlib visualization
    points = np.asarray(pcd_blend.points)
    colors = np.asarray(pcd_blend.colors)
    
    # Downsample
    max_points = 50000
    if len(points) > max_points:
        step = len(points) // max_points
        points_viz = points[::step]
        colors_viz = colors[::step]
    else:
        points_viz = points
        colors_viz = colors
    
    fig = plt.figure(figsize=(20, 15))
    
    viewpoints = [
        (1, "Front View", 0, 0),
        (2, "Top View", 90, 0),
        (3, "Side View", 0, 90),
        (4, "Angled View", 30, 45),
    ]
    
    # Build legend elements if metadata available
    legend_elements = []
    if segments_info is not None and metadata is not None:
        unique_categories = set()
        for seg in segments_info:
            unique_categories.add(seg['category_id'])
        
        class_colors_used = set()
        for cat_id in sorted(unique_categories):
            if cat_id < len(metadata.stuff_classes):
                class_name = metadata.stuff_classes[cat_id]
            elif hasattr(metadata, 'thing_classes') and (cat_id - len(metadata.stuff_classes)) < len(metadata.thing_classes):
                class_name = metadata.thing_classes[cat_id - len(metadata.stuff_classes)]
            else:
                class_name = f"Class {cat_id}"
            
            # Get color for this class (blend with alpha)
            if hasattr(metadata, 'stuff_colors') and cat_id < len(metadata.stuff_colors):
                sem_color = np.array(metadata.stuff_colors[cat_id]) / 255.0
            else:
                sem_color = plt.cm.tab20(cat_id % 20)[:3]
            
            # Approximate blended color (assume average RGB is gray)
            blended_color = (1 - alpha) * np.array([0.5, 0.5, 0.5]) + alpha * sem_color
            
            color_tuple = tuple(blended_color)
            if color_tuple not in class_colors_used:
                class_colors_used.add(color_tuple)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=blended_color, markersize=10,
                                                  label=class_name))
    
    for idx, title, elev, azim in viewpoints:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2],
                  c=colors_viz, s=1, alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"{title}\n(RGB + Semantic Blend, α={alpha})", 
                    fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio
        max_range = np.array([
            points_viz[:, 0].max() - points_viz[:, 0].min(),
            points_viz[:, 1].max() - points_viz[:, 1].min(),
            points_viz[:, 2].max() - points_viz[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points_viz[:, 0].max() + points_viz[:, 0].min()) * 0.5
        mid_y = (points_viz[:, 1].max() + points_viz[:, 1].min()) * 0.5
        mid_z = (points_viz[:, 2].max() + points_viz[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add legend to the first subplot
        if idx == 1 and legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 1),
                     fontsize=8, framealpha=0.9, title="Classes")
    
    plt.tight_layout()
    
    blend_viz_path = os.path.join(output_dir, "pointcloud_blended_views.png")
    plt.savefig(blend_viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Blended visualization saved: {blend_viz_path}")
    plt.close()


def render_point_cloud_images(pcd, output_dir, segments_info=None, metadata=None):
    """
    Render point cloud from multiple viewpoints as images (headless rendering).
    Alternative to interactive visualization for cluster environments.
    
    Args:
        pcd: Open3D point cloud
        output_dir: Directory to save outputs
        segments_info: Optional segment metadata for adding class labels
        metadata: Optional COCO metadata for class names
    """
    print("\nRendering point cloud views (headless mode)...")
    
    try:
        # Create visualizer for offscreen rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        vis.add_geometry(pcd)
        
        # Add 3D text labels if metadata is available
        text_geometries = []
        if segments_info is not None and metadata is not None:
            print("  Adding 3D class labels to visualization...")
            
            # Get unique categories
            unique_categories = set()
            for seg in segments_info:
                unique_categories.add(seg['category_id'])
            
            # Get bounding box to position labels
            bbox = pcd.get_axis_aligned_bounding_box()
            min_bound = bbox.min_bound
            max_bound = bbox.max_bound
            
            # Position labels in a column on the left side
            label_x = min_bound[0] - (max_bound[0] - min_bound[0]) * 0.3
            label_z_start = max_bound[2]
            label_spacing = (max_bound[2] - min_bound[2]) / (len(unique_categories) + 1)
            
            for idx, cat_id in enumerate(sorted(unique_categories)):
                # Get class name
                if cat_id < len(metadata.stuff_classes):
                    class_name = metadata.stuff_classes[cat_id]
                elif hasattr(metadata, 'thing_classes') and (cat_id - len(metadata.stuff_classes)) < len(metadata.thing_classes):
                    class_name = metadata.thing_classes[cat_id - len(metadata.stuff_classes)]
                else:
                    class_name = f"Class {cat_id}"
                
                # Get color for this class
                if hasattr(metadata, 'stuff_colors') and cat_id < len(metadata.stuff_colors):
                    color = np.array(metadata.stuff_colors[cat_id]) / 255.0
                else:
                    color = plt.cm.tab20(cat_id % 20)[:3]
                
                # Create 3D text label
                label_position = [label_x, min_bound[1], label_z_start - idx * label_spacing]
                
                # Create a small sphere as color indicator
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(label_position)
                sphere.paint_uniform_color(color)
                vis.add_geometry(sphere)
                text_geometries.append(sphere)
                
                # Note: Open3D doesn't support text rendering in headless mode well
                # So we'll overlay text using matplotlib after rendering
            
            print(f"  Added {len(unique_categories)} class indicators")
        
        # Get render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Define viewpoints (azimuth, elevation, distance)
        viewpoints = [
            ("front", 0, 0, 2.5),
            ("top", 0, 90, 2.5),
            ("side", 90, 0, 2.5),
            ("angle", 45, 30, 2.5),
        ]
        
        for name, azimuth, elevation, distance in viewpoints:
            # Set camera viewpoint
            ctr = vis.get_view_control()
            ctr.set_zoom(0.5)
            ctr.rotate(azimuth * 10, elevation * 10)
            
            # Render and save
            vis.poll_events()
            vis.update_renderer()
            
            output_path = os.path.join(output_dir, f"pointcloud_{name}.png")
            vis.capture_screen_image(output_path, do_render=True)
            print(f"  ✓ {name} view saved: {output_path}")
            
            # Overlay text labels using matplotlib if metadata available
            if segments_info is not None and metadata is not None:
                add_text_overlay_to_image(output_path, segments_info, metadata)
        
        vis.destroy_window()
        print("✓ Point cloud renderings complete")
        return True
        
    except Exception as e:
        print(f"⚠ Headless rendering failed: {e}")
        print("  Point cloud saved as .ply file - download and view locally with:")
        print("  - MeshLab (https://www.meshlab.net/)")
        print("  - CloudCompare (https://www.cloudcompare.org/)")
        print("  - Python: import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('reconstruction.ply')])")
        return False


def add_text_overlay_to_image(image_path, segments_info, metadata):
    """
    Add text overlay with class labels to rendered Open3D image
    
    Args:
        image_path: Path to the rendered image
        segments_info: Segment metadata with category IDs
        metadata: COCO metadata for class names
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    # Load image
    img = PILImage.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Get unique categories
    unique_categories = set()
    for seg in segments_info:
        unique_categories.add(seg['category_id'])
    
    # Create legend in top-left corner
    legend_x = 20
    legend_y = 20
    line_height = 30
    
    # Draw semi-transparent background for legend
    legend_height = len(unique_categories) * line_height + 40
    legend_width = 250
    draw.rectangle(
        [(legend_x - 10, legend_y - 10), (legend_x + legend_width, legend_y + legend_height)],
        fill=(0, 0, 0, 200)
    )
    
    # Draw title
    draw.text((legend_x, legend_y), "Classes:", fill=(255, 255, 255), font=font)
    legend_y += 35
    
    # Draw each class with color box
    for cat_id in sorted(unique_categories):
        # Get class name
        if cat_id < len(metadata.stuff_classes):
            class_name = metadata.stuff_classes[cat_id]
        elif hasattr(metadata, 'thing_classes') and (cat_id - len(metadata.stuff_classes)) < len(metadata.thing_classes):
            class_name = metadata.thing_classes[cat_id - len(metadata.stuff_classes)]
        else:
            class_name = f"Class {cat_id}"
        
        # Get color for this class
        if hasattr(metadata, 'stuff_colors') and cat_id < len(metadata.stuff_colors):
            color = tuple(metadata.stuff_colors[cat_id])
        else:
            color_normalized = plt.cm.tab20(cat_id % 20)[:3]
            color = tuple(int(c * 255) for c in color_normalized)
        
        # Draw color box
        box_size = 20
        draw.rectangle(
            [(legend_x, legend_y), (legend_x + box_size, legend_y + box_size)],
            fill=color,
            outline=(255, 255, 255),
            width=2
        )
        
        # Draw class name
        draw.text((legend_x + box_size + 10, legend_y + 2), class_name, fill=(255, 255, 255), font=font_small)
        
        legend_y += line_height
    
    # Save modified image
    img.save(image_path)
    print(f"    Added text overlay with {len(unique_categories)} class labels")


def create_matplotlib_3d_view(pts3d, confidence, image, output_dir, conf_threshold=0.5):
    """
    Create 3D scatter plot visualization using matplotlib (works on cluster).
    Alternative to Open3D interactive visualization.
    """
    print("\nCreating matplotlib 3D visualization...")
    
    H, W = pts3d.shape[:2]
    
    # Flatten and filter
    points = pts3d.reshape(-1, 3)
    confidences = confidence.reshape(-1)
    valid_mask = confidences > conf_threshold
    points_filtered = points[valid_mask]
    
    # Get colors
    img_resized = np.array(image.resize((W, H)))
    colors = img_resized.reshape(-1, 3) / 255.0
    colors_filtered = colors[valid_mask]
    
    # Downsample for faster rendering (matplotlib is slow with many points)
    step = max(1, len(points_filtered) // 50000)  # Max 50k points for visualization
    points_viz = points_filtered[::step]
    colors_viz = colors_filtered[::step]
    
    # Create figure with multiple viewpoints
    fig = plt.figure(figsize=(20, 15))
    
    viewpoints = [
        (1, "Front View (XY)", 0, 0),
        (2, "Top View (XZ)", 90, 0),
        (3, "Side View (YZ)", 0, 90),
        (4, "Angled View", 45, 30),
    ]
    
    for idx, title, elev, azim in viewpoints:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        ax.scatter(
            points_viz[:, 0], 
            points_viz[:, 1], 
            points_viz[:, 2],
            c=colors_viz,
            s=1,
            alpha=0.6
        )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio
        max_range = np.array([
            points_viz[:, 0].max() - points_viz[:, 0].min(),
            points_viz[:, 1].max() - points_viz[:, 1].min(),
            points_viz[:, 2].max() - points_viz[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points_viz[:, 0].max() + points_viz[:, 0].min()) * 0.5
        mid_y = (points_viz[:, 1].max() + points_viz[:, 1].min()) * 0.5
        mid_z = (points_viz[:, 2].max() + points_viz[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "pointcloud_3d_views.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 3D visualization saved: {output_path}")
    print(f"  (Showing {len(points_viz):,} of {len(points_filtered):,} points)")
    
    plt.close()
    return output_path


# ============================================================
# MASK2FORMER SEGMENTATION
# ============================================================

def run_mask2former_inference(image_path, model, cfg):
    """Run Mask2Former for panoptic segmentation"""
    print("\n" + "="*80)
    print("MASK2FORMER PANOPTIC SEGMENTATION")
    print("="*80)
    
    # Load image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    print(f"Original image size: {width}x{height}")
    
    # Use standard test-time augmentation
    aug = T.ResizeShortestEdge([800, 800], 1333)
    
    image = aug.get_transform(img).apply_image(img)
    print(f"Resized for inference: {image.shape[1]}x{image.shape[0]}")
    
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    
    inputs = {"image": image, "height": height, "width": width}
    
    print("Running Mask2Former inference...")
    with torch.no_grad():
        predictions = model([inputs])[0]
    
    print(f"✓ Segmentation complete")
    
    return predictions, img


def visualize_segmentation(img, predictions, cfg, output_path):
    """Visualize panoptic segmentation results"""
    # Fix numpy compatibility issue with detectron2
    import numpy as np
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    
    # Get metadata - need to create a mutable copy since MetadataCatalog is immutable
    from detectron2.data import Metadata
    metadata_orig = MetadataCatalog.get("coco_2017_val_panoptic")
    
    # Create a new metadata with converted colors
    metadata = Metadata(name="coco_2017_val_panoptic_converted")
    
    # Copy all attributes except colors first
    for key, value in metadata_orig.as_dict().items():
        if key not in ['stuff_colors', 'thing_colors', 'name']:
            try:
                setattr(metadata, key, value)
            except:
                pass  # Skip immutable attributes
    
    # Handle color conversion separately
    if hasattr(metadata_orig, 'stuff_colors') and metadata_orig.stuff_colors:
        stuff_colors = metadata_orig.stuff_colors
        if stuff_colors and len(stuff_colors) > 0 and max(stuff_colors[0]) > 1.0:
            # Convert from 0-255 to 0-1 range
            metadata.stuff_colors = [[c/255.0 for c in color] for color in stuff_colors]
        else:
            metadata.stuff_colors = stuff_colors
    
    if hasattr(metadata_orig, 'thing_colors') and metadata_orig.thing_colors:
        thing_colors = metadata_orig.thing_colors
        if thing_colors and len(thing_colors) > 0 and max(thing_colors[0]) > 1.0:
            # Convert from 0-255 to 0-1 range
            metadata.thing_colors = [[c/255.0 for c in color] for color in thing_colors]
        else:
            metadata.thing_colors = thing_colors
    
    # Create visualizer
    visualizer = Visualizer(
        img[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        instance_mode=ColorMode.IMAGE
    )
    
    # Draw predictions
    if "panoptic_seg" in predictions:
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg(
            panoptic_seg.to("cpu"),
            segments_info
        )
    elif "sem_seg" in predictions:
        vis_output = visualizer.draw_sem_seg(
            predictions["sem_seg"].argmax(dim=0).to("cpu")
        )
    else:
        print("Warning: No segmentation output found!")
        return None
    
    # Save visualization
    vis_output.save(output_path)
    print(f"✓ Segmentation visualization saved: {output_path}")
    
    return vis_output.get_image()


# ============================================================
# COMBINED VISUALIZATION
# ============================================================

def create_combined_visualization(image_path, mapanything_results, seg_image, output_dir):
    """Create combined visualization of all results"""
    print("\n" + "="*80)
    print("CREATING COMBINED VISUALIZATION")
    print("="*80)
    
    # Load original image
    img = Image.open(image_path).convert('RGB')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Depth map
    ax2 = plt.subplot(2, 3, 2)
    depth_vis = ax2.imshow(mapanything_results['metric_depth'], cmap='plasma')
    ax2.set_title(f"Metric Depth\n{mapanything_results['metric_depth'].min():.1f}m - {mapanything_results['metric_depth'].max():.1f}m", 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(depth_vis, ax=ax2, label='Depth (meters)', fraction=0.046)
    
    # Confidence map
    ax3 = plt.subplot(2, 3, 3)
    conf_vis = ax3.imshow(mapanything_results['confidence'], cmap='viridis')
    ax3.set_title("Confidence Map", fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(conf_vis, ax=ax3, label='Confidence', fraction=0.046)
    
    # Segmentation result
    ax4 = plt.subplot(2, 3, 4)
    if seg_image is not None:
        ax4.imshow(seg_image)
    ax4.set_title("Panoptic Segmentation", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Depth histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(mapanything_results['metric_depth'].flatten(), bins=50, color='skyblue', edgecolor='black')
    ax5.set_title("Depth Distribution", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Depth (meters)")
    ax5.set_ylabel("Frequency")
    ax5.grid(True, alpha=0.3)
    
    # Statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    3D RECONSTRUCTION STATISTICS
    {'='*40}
    
    Depth Range: {float(mapanything_results['metric_depth'].min()):.2f}m - {float(mapanything_results['metric_depth'].max()):.2f}m
    Mean Depth: {float(mapanything_results['metric_depth'].mean()):.2f}m
    Metric Scale: {mapanything_results['metric_scale']:.3f}
    
    Confidence Range: {float(mapanything_results['confidence'].min()):.3f} - {float(mapanything_results['confidence'].max()):.3f}
    Mean Confidence: {float(mapanything_results['confidence'].mean()):.3f}
    
    Camera Translation: 
      [{float(mapanything_results['cam_trans'].flatten()[0]):.3f}, 
       {float(mapanything_results['cam_trans'].flatten()[1]):.3f}, 
       {float(mapanything_results['cam_trans'].flatten()[2]):.3f}]
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
             verticalalignment='center')
    
    plt.tight_layout()
    
    # Save combined visualization
    output_path = os.path.join(output_dir, "combined_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Combined visualization saved: {output_path}")
    
    plt.close()


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def run_inference(
    image_path,
    checkpoint_path,
    output_dir="./inference_output",
    conf_threshold=0.5,
    save_pointcloud=True,
    visualize_3d=False
):
    """
    Run unified inference pipeline - both segmentation and 3D reconstruction together
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save outputs
        conf_threshold: Confidence threshold for point cloud filtering
        save_pointcloud: Whether to save point cloud as PLY file
        visualize_3d: Whether to show interactive 3D visualization
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*80)
    print("MASK2FORMER + MAPANYTHING UNIFIED INFERENCE")
    print("="*80)
    print(f"Input image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Load and prepare image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    print(f"\nOriginal image size: {width}x{height}")
    
    # Use standard test-time image sizing
    aug = T.ResizeShortestEdge([800, 800], 1333)
    image_resized = aug.get_transform(img).apply_image(img)
    print(f"Resized for inference: {image_resized.shape[1]}x{image_resized.shape[0]}")
    
    image_tensor = torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1))
    
    # ============================================================
    # UNIFIED INFERENCE: Run both tasks together
    # ============================================================
    print("\n" + "="*80)
    print("RUNNING UNIFIED INFERENCE")
    print("="*80)
    
    # Load model
    model, cfg = load_model(checkpoint_path)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Prepare inputs for both tasks
    inputs = {"image": image_tensor, "height": height, "width": width}
    
    # Prepare MapAnything input format (supports single image)
    views = load_images([image_path])
    
    # Move views to device
    mapanything_model = model.backbone.mapanything
    ma_device = next(mapanything_model.parameters()).device
    for view in views:
        for key, val in view.items():
            if isinstance(val, torch.Tensor):
                view[key] = val.to(ma_device)
    
    with torch.no_grad():
        # Task 1: Panoptic Segmentation
        print("Running panoptic segmentation...")
        predictions = model([inputs])[0]
        print("✓ Segmentation complete")
        
        # Task 2: 3D Reconstruction (using same backbone)
        print("Running 3D reconstruction...")
        mapanything_predictions = mapanything_model(views)
        print("✓ 3D reconstruction complete")
    
    # ============================================================
    # PROCESS SEGMENTATION RESULTS
    # ============================================================
    print("\n" + "="*80)
    print("PROCESSING SEGMENTATION RESULTS")
    print("="*80)
    
    seg_output_path = os.path.join(output_dir, "segmentation.png")
    seg_image = visualize_segmentation(img, predictions, cfg, seg_output_path)
    
    # ============================================================
    # PROCESS 3D RECONSTRUCTION RESULTS
    # ============================================================
    print("\n" + "="*80)
    print("PROCESSING 3D RECONSTRUCTION RESULTS")
    print("="*80)
    
    pred = mapanything_predictions[0]
    
    mapanything_results = {
        'depth': pred['depth_along_ray'].cpu().numpy().squeeze(),
        'metric_scale': pred['metric_scaling_factor'].item() if isinstance(pred['metric_scaling_factor'], torch.Tensor) else pred['metric_scaling_factor'],
        'pts3d': pred['pts3d'].cpu().numpy().squeeze(),
        'confidence': pred['conf'].cpu().numpy().squeeze(),
        'cam_trans': pred['cam_trans'].cpu().numpy(),
        'cam_quats': pred['cam_quats'].cpu().numpy(),
    }
    mapanything_results['metric_depth'] = mapanything_results['depth'] * mapanything_results['metric_scale']
    
    print(f"  Depth range: {mapanything_results['metric_depth'].min():.2f}m - {mapanything_results['metric_depth'].max():.2f}m")
    print(f"  Metric scale: {mapanything_results['metric_scale']:.3f}")
    print(f"  DEBUG: pts3d shape: {mapanything_results['pts3d'].shape}")
    print(f"  DEBUG: depth shape: {mapanything_results['metric_depth'].shape}")
    
    # Clean up model from memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============================================================
    # SAVE ALL OUTPUTS
    # ============================================================
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save depth map
    depth_path = os.path.join(output_dir, "depth_map.npy")
    np.save(depth_path, mapanything_results['metric_depth'])
    print(f"✓ Depth map saved: {depth_path}")
    
    # Create and save point clouds (RGB + Semantic)
    img_pil = Image.open(image_path).convert('RGB')
    
    # Extract segmentation mask from predictions
    seg_mask = None
    segments_info = None
    if "panoptic_seg" in predictions:
        seg_mask = predictions["panoptic_seg"][0].cpu().numpy()  # Get panoptic mask
        segments_info = predictions["panoptic_seg"][1]  # Get segment metadata (id -> category_id mapping)
        print(f"  DEBUG: Segmentation mask shape: {seg_mask.shape}")
    elif "sem_seg" in predictions:
        seg_mask = predictions["sem_seg"].argmax(dim=0).cpu().numpy()  # Get semantic mask
        print(f"  DEBUG: Segmentation mask shape: {seg_mask.shape}")
    
    # Get metadata for class colors
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    
    print(f"  DEBUG: Original image shape (H, W): {height}x{width}")
    print(f"  DEBUG: pts3d shape (H, W, 3): {mapanything_results['pts3d'].shape}")
    print(f"  DEBUG: Segmentation mask shape: {seg_mask.shape if seg_mask is not None else 'None'}")
    print(f"")
    print(f"  RESOLUTION MISMATCH EXPLANATION:")
    print(f"  - Original image: {width}×{height}")
    print(f"  - Segmentation mask: {seg_mask.shape if seg_mask is not None else 'N/A'} (returned to original size by Mask2Former)")
    print(f"  - pts3d from MapAnything: {mapanything_results['pts3d'].shape[:2]}")
    print(f"    (MapAnything uses load_images with resolution_set=518 by default,")
    print(f"     which applies fixed_mapping resize to ~518×518 or similar based on aspect ratio)")
    print(f"  - Solution: Resize segmentation mask to match pts3d resolution using NEAREST interpolation")
    print(f"")
    
    pcd_rgb, pcd_semantic = create_point_cloud(
        mapanything_results['pts3d'],
        mapanything_results['confidence'],
        img_pil,
        conf_threshold=conf_threshold,
        segmentation_mask=seg_mask,
        segments_info=segments_info,
        metadata=metadata
    )
    
    if save_pointcloud:
        # Save RGB point cloud
        pcd_rgb_path = os.path.join(output_dir, "reconstruction_rgb.ply")
        o3d.io.write_point_cloud(pcd_rgb_path, pcd_rgb)
        print(f"✓ RGB point cloud saved: {pcd_rgb_path}")
        
        # Save semantic point cloud if available
        if pcd_semantic is not None:
            pcd_semantic_path = os.path.join(output_dir, "reconstruction_semantic.ply")
            o3d.io.write_point_cloud(pcd_semantic_path, pcd_semantic)
            print(f"✓ Semantic point cloud saved: {pcd_semantic_path}")
            print(f"  (Points colored by predicted semantic class)")
            
            # Create side-by-side comparison visualization
            print(f"\n  Creating comparison visualizations...")
            create_point_cloud_comparison(
                pcd_rgb, 
                pcd_semantic, 
                output_dir,
                mapanything_results['pts3d'],
                mapanything_results['confidence'],
                seg_mask,
                segments_info,
                metadata,
                conf_threshold
            )
    
    # Use RGB point cloud for remaining visualizations
    pcd = pcd_rgb
    
    print("\n" + "="*80)
    print("✓ UNIFIED INFERENCE COMPLETE!")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")
    print(f"  - segmentation.png (panoptic segmentation overlay)")
    print(f"  - depth_map.npy (metric depth array)")
    print(f"  - reconstruction_rgb.ply (3D point cloud with RGB colors)")
    print(f"  - reconstruction_semantic.ply (3D point cloud with semantic class colors)")
    print(f"  - reconstruction_blended.ply (RGB + Semantic overlay point cloud)")
    print(f"  - combined_results.png (all metrics and visualizations)")
    print(f"  - pointcloud_3d_views.png (RGB point cloud from 4 angles)")
    print(f"  - pointcloud_rgb_vs_semantic.png (Side-by-side comparison with labels)")
    print(f"  - pointcloud_blended_views.png (Blended RGB+Semantic from 4 angles)")
    if visualize_3d:
        print(f"  - pointcloud_*.png (rendered views from Open3D)")
    print("="*80)
    create_matplotlib_3d_view(
        mapanything_results['pts3d'],
        mapanything_results['confidence'],
        img_pil,
        output_dir,
        conf_threshold=conf_threshold
    )
    
    # Try headless rendering with Open3D
    if visualize_3d:
        render_point_cloud_images(pcd, output_dir, segments_info, metadata)
    
    print("\n" + "="*80)
    print("✓ UNIFIED INFERENCE COMPLETE!")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")
    print(f"  - segmentation.png (panoptic segmentation overlay)")
    print(f"  - depth_map.npy (metric depth array)")
    print(f"  - reconstruction.ply (3D point cloud - download to view)")
    print(f"  - combined_results.png (all metrics and visualizations)")
    print(f"  - pointcloud_3d_views.png (3D point cloud from 4 angles)")
    if visualize_3d:
        print(f"  - pointcloud_*.png (rendered views from Open3D)")
    print("\n💡 To view the .ply point clouds interactively on your LOCAL machine:")
    print("")
    print("   View RGB point cloud (original image colors):")
    print(f"   python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('reconstruction_rgb.ply')])\"")
    print("")
    print("   View SEMANTIC point cloud (colored by predicted class):")
    print(f"   python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('reconstruction_semantic.ply')])\"")
    print("")
    print("   View BLENDED overlay (RGB + Semantic):")
    print(f"   python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('reconstruction_blended.ply')])\"")
    print("")
    print("   Compare RGB vs Semantic side-by-side:")
    print("   " + "-"*60)
    print("   import open3d as o3d")
    print("   pcd_rgb = o3d.io.read_point_cloud('reconstruction_rgb.ply')")
    print("   pcd_sem = o3d.io.read_point_cloud('reconstruction_semantic.ply')")
    print("   pcd_sem.translate([3, 0, 0])  # Offset for side-by-side view")
    print("   o3d.visualization.draw_geometries([pcd_rgb, pcd_sem])")
    print("   " + "-"*60)
    print("")
    print("   PNG visualizations (no Open3D needed):")
    print("   • pointcloud_rgb_vs_semantic.png - Side-by-side comparison with class labels")
    print("   • pointcloud_blended_views.png - RGB+Semantic overlay from 4 angles")
    print("   • pointcloud_3d_views.png - RGB point cloud from 4 angles")
    print("")
    print("   Other free 3D viewers:")
    print("   • MeshLab: https://www.meshlab.net/")
    print("   • CloudCompare: https://www.cloudcompare.org/")
    return {
        'mapanything': mapanything_results,
        'segmentation': predictions,
        'pointcloud_rgb': pcd_rgb,
        'pointcloud_semantic': pcd_semantic
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with Mask2Former + MapAnything"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./output_cluster/model_final.pth",
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./inference_output",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for point cloud filtering (0-1)"
    )
    
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Skip saving point cloud"
    )
    
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Try headless Open3D rendering (may fail on cluster without display)"
    )
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        save_pointcloud=not args.no_pointcloud,
        visualize_3d=args.visualize_3d
    )
