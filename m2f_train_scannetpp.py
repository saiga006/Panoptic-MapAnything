"""
ScanNet++ Training Script for MapAnything + Mask2Former Panoptic Segmentation

This script adapts the COCO training pipeline for ScanNet++ dataset with:
1. Multi-view support for MapAnything backbone
2. Camera pose integration for 3D-aware feature extraction
3. View selection with minimal overlap for diverse training

ScanNet++ Dataset Structure:
============================
The dataset contains high-fidelity 3D indoor scenes with:
- DSLR images: High-resolution fisheye images (can be undistorted)
- iPhone captures: RGB + LiDAR depth at 60fps
- 3D meshes with semantic/instance annotations
- Camera poses in metric scale (COLMAP format)

Training Approach:
==================
1. Single Scene Training: Train on one scene with diverse views
   - Use farthest point sampling to select views with minimal overlap
   - Ensure coverage of different room areas
   
2. Multi-Scene Training: Train across multiple scenes
   - Sample views from different scenes each iteration
   - Use camera poses for multi-view fusion in MapAnything

Usage:
------
# Single scene training
python m2f_train_scannetpp.py \\
    --mapanything-checkpoint /path/to/mapanything.pt \\
    --scannetpp-root /path/to/scannetpp/data \\
    --scene-id abc123 \\
    --output-dir ./output_scannetpp

# Multi-scene training
python m2f_train_scannetpp.py \\
    --mapanything-checkpoint /path/to/mapanything.pt \\
    --scannetpp-root /path/to/scannetpp/data \\
    --split-file /path/to/nvs_sem_train.txt \\
    --output-dir ./output_scannetpp
"""

import os
import sys
import copy
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Filter FutureWarnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# Detectron2 imports
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, default_argument_parser
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm

# Ensure local mask2former is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask2former import add_maskformer2_config
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    ScanNetPPPanopticDatasetMapper,
    ScanNetPPMultiViewDatasetMapper,
    ScanNetPPScene,
    read_nerfstudio_transforms,
    select_views_for_scene,
    register_scannetpp_panoptic,
)

# Import backbone from main training script
from m2f_train_3d_loss import (
    MapAnythingWithPanopticDPT,
    Mask2FormerPanopticTrainer,
    NaNLossCheckHook,
)


# ============================================================
# SCANNET++ SPECIFIC CONFIGURATION
# ============================================================

def add_scannetpp_config(cfg):
    """Add ScanNet++ specific configuration options."""
    # Dataset configuration
    cfg.DATASETS.SCANNETPP_ROOT = ""
    cfg.DATASETS.SCANNETPP_SPLIT = ""
    cfg.DATASETS.SCANNETPP_SCENE_ID = ""  # For single-scene training
    cfg.DATASETS.SCANNETPP_PANOPTIC_DIR = ""  # 2D panoptic annotations
    
    # Input configuration for multi-view
    cfg.INPUT.IMAGE_TYPE = "dslr"  # 'dslr' or 'iphone'
    cfg.INPUT.USE_UNDISTORTED = True
    cfg.INPUT.NUM_VIEWS = 4  # Number of views per sample
    cfg.INPUT.MIN_VIEW_DISTANCE = 0.5  # Minimum distance (m) between views
    cfg.INPUT.SIZE_DIVISIBILITY = 32
    
    # ScanNet++ has 100 semantic classes in benchmark
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 100


# ============================================================
# SCANNET++ DATASET REGISTRATION
# ============================================================

def register_scannetpp_single_scene(
    scene_id: str,
    data_root: str,
    panoptic_dir: Optional[str] = None,
    image_type: str = 'dslr',
    use_undistorted: bool = True,
    num_views_train: int = -1,  # -1 means use all
    num_views_val: int = -1,
    min_view_distance: float = 0.5,
):
    """
    Register a single ScanNet++ scene for training.
    
    This is useful for:
    - Debugging on a single scene
    - Scene-specific fine-tuning
    - Memory-constrained training
    
    Args:
        scene_id: ScanNet++ scene ID
        data_root: Root of ScanNet++ data directory
        panoptic_dir: Directory with 2D panoptic annotations (optional)
        image_type: 'dslr' or 'iphone'
        use_undistorted: Use undistorted pinhole images
        num_views_train: Number of views for training (-1 = all)
        num_views_val: Number of views for validation (-1 = all)
        min_view_distance: Minimum camera distance for view selection
    """
    scene = ScanNetPPScene(scene_id, data_root)
    
    # Get transforms file path
    if image_type == 'dslr':
        transforms_path = (
            scene.dslr_nerfstudio_transforms_undistorted if use_undistorted
            else scene.dslr_nerfstudio_transforms
        )
        image_dir = (
            scene.dslr_undistorted_dir if use_undistorted
            else scene.dslr_resized_dir
        )
    else:
        transforms_path = scene.iphone_nerfstudio_transforms
        image_dir = scene.iphone_rgb_dir
    
    # Load camera data
    intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
    
    # Select diverse views for training
    if num_views_train > 0:
        train_indices = select_views_for_scene(
            frames, num_views_train, min_view_distance, seed=42
        )
    else:
        train_indices = list(range(len(frames)))
    
    # Use different seed for validation views
    if num_views_val > 0:
        val_indices = select_views_for_scene(
            frames, num_views_val, min_view_distance, seed=123
        )
    else:
        val_indices = list(range(len(frames)))
    
    def create_dataset_dicts(indices, split='train'):
        dataset_dicts = []
        for i, idx in enumerate(indices):
            frame = frames[idx]
            file_path = frame.get('file_path', '')
            
            # Resolve image path
            if file_path:
                if not Path(file_path).is_absolute():
                    img_name = Path(file_path).name
                    file_path = str(image_dir / img_name)
            else:
                # Fallback: list images in directory
                images = sorted(image_dir.glob('*.JPG')) + sorted(image_dir.glob('*.jpg'))
                if idx < len(images):
                    file_path = str(images[idx])
            
            record = {
                'image_id': i,
                'scene_id': scene_id,
                'file_name': file_path,
                'camera_to_world': frame['camera_to_world'],
                'intrinsics': intrinsics,
            }
            
            # Add dimensions from intrinsics
            record['height'] = intrinsics.get('h', 1440)
            record['width'] = intrinsics.get('w', 1920)
            
            # Add panoptic annotation if available
            if panoptic_dir:
                stem = Path(file_path).stem
                pan_path = Path(panoptic_dir) / scene_id / f"{stem}.png"
                if pan_path.exists():
                    record['pan_seg_file_name'] = str(pan_path)
                    
                    info_path = Path(panoptic_dir) / scene_id / f"{stem}.json"
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            record['segments_info'] = json.load(f)
                    else:
                        record['segments_info'] = []
            
            dataset_dicts.append(record)
        
        return dataset_dicts
    
    # Register training set
    train_name = f"scannetpp_{scene_id}_train"
    if train_name not in DatasetCatalog:
        DatasetCatalog.register(
            train_name,
            lambda: create_dataset_dicts(train_indices, 'train')
        )
        MetadataCatalog.get(train_name).set(
            scene_id=scene_id,
            image_type=image_type,
            num_views=len(train_indices),
            ignore_label=255,
        )
    
    # Register validation set
    val_name = f"scannetpp_{scene_id}_val"
    if val_name not in DatasetCatalog:
        DatasetCatalog.register(
            val_name,
            lambda: create_dataset_dicts(val_indices, 'val')
        )
        MetadataCatalog.get(val_name).set(
            scene_id=scene_id,
            image_type=image_type,
            num_views=len(val_indices),
            ignore_label=255,
        )
    
    print(f"\n{'='*60}")
    print(f"REGISTERED SCANNET++ SCENE: {scene_id}")
    print(f"{'='*60}")
    print(f"  Image type: {image_type} (undistorted={use_undistorted})")
    print(f"  Total frames: {len(frames)}")
    print(f"  Training views: {len(train_indices)}")
    print(f"  Validation views: {len(val_indices)}")
    print(f"  Min view distance: {min_view_distance}m")
    print(f"  Image directory: {image_dir}")
    print(f"{'='*60}\n")
    
    return train_name, val_name


# ============================================================
# SCANNET++ TRAINER
# ============================================================

class ScanNetPPTrainer(Mask2FormerPanopticTrainer):
    """
    Custom trainer for ScanNet++ with multi-view support.
    
    Extends Mask2FormerPanopticTrainer with:
    - ScanNet++ specific data loading
    - Multi-view batch handling
    - Camera pose integration
    """
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader with ScanNet++ mapper."""
        
        # Choose mapper based on training mode
        if cfg.INPUT.NUM_VIEWS > 1:
            # Multi-view mode
            mapper = ScanNetPPMultiViewDatasetMapper(
                is_train=True,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=(640, 672, 704, 736, 768),
                        max_size=1333,
                        sample_style="choice"
                    ),
                ],
                image_format=cfg.INPUT.FORMAT,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                num_views=cfg.INPUT.NUM_VIEWS,
                min_view_distance=cfg.INPUT.MIN_VIEW_DISTANCE,
                use_undistorted=cfg.INPUT.USE_UNDISTORTED,
                image_type=cfg.INPUT.IMAGE_TYPE,
                data_root=cfg.DATASETS.SCANNETPP_ROOT,
            )
        else:
            # Single-view mode
            mapper = ScanNetPPPanopticDatasetMapper(
                is_train=True,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=(640, 672, 704, 736, 768),
                        max_size=1333,
                        sample_style="choice"
                    ),
                ],
                image_format=cfg.INPUT.FORMAT,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
                use_undistorted=cfg.INPUT.USE_UNDISTORTED,
                image_type=cfg.INPUT.IMAGE_TYPE,
            )
        
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test data loader."""
        mapper = ScanNetPPPanopticDatasetMapper(
            is_train=False,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=800,
                    max_size=1333,
                    sample_style="choice"
                ),
            ],
            image_format=cfg.INPUT.FORMAT,
            ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY,
            use_undistorted=cfg.INPUT.USE_UNDISTORTED,
            image_type=cfg.INPUT.IMAGE_TYPE,
        )
        
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


# ============================================================
# CONFIGURATION SETUP
# ============================================================

def setup_scannetpp_cfg(
    mapanything_checkpoint: str,
    scannetpp_root: str,
    output_dir: str,
    scene_id: Optional[str] = None,
    split_file: Optional[str] = None,
    panoptic_dir: Optional[str] = None,
    image_type: str = 'dslr',
    use_undistorted: bool = True,
    num_views: int = 1,
    min_view_distance: float = 0.5,
    num_gpus: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    dpt_lr: float = 1e-5,
    max_iter: int = 10000,
):
    """
    Setup configuration for ScanNet++ training.
    
    Args:
        mapanything_checkpoint: Path to MapAnything checkpoint
        scannetpp_root: Root directory of ScanNet++ data
        output_dir: Output directory for checkpoints and logs
        scene_id: Single scene ID for training (optional)
        split_file: Path to split file for multi-scene training (optional)
        panoptic_dir: Directory with 2D panoptic annotations
        image_type: 'dslr' or 'iphone'
        use_undistorted: Use undistorted images
        num_views: Number of views per sample (1 = single-view)
        min_view_distance: Minimum distance between selected views
        num_gpus: Number of GPUs
        batch_size: Batch size per GPU
        learning_rate: Base learning rate
        dpt_lr: Learning rate for DPT components
        max_iter: Maximum training iterations
    
    Returns:
        Detectron2 config node
    """
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_scannetpp_config(cfg)
    
    # ===== MODEL ARCHITECTURE =====
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # ===== BACKBONE CONFIGURATION =====
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_panoptic_dpt_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    cfg.MODEL.MAPANYTHING = CN()
    cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH = mapanything_checkpoint
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # ===== MASK2FORMER HEAD =====
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 100  # ScanNet++ benchmark classes
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    
    # ===== PIXEL DECODER =====
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    
    # ===== MASK FORMER DECODER =====
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
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
    
    # ===== SCANNET++ SPECIFIC =====
    cfg.DATASETS.SCANNETPP_ROOT = scannetpp_root
    cfg.DATASETS.SCANNETPP_SCENE_ID = scene_id or ""
    cfg.DATASETS.SCANNETPP_SPLIT = split_file or ""
    cfg.DATASETS.SCANNETPP_PANOPTIC_DIR = panoptic_dir or ""
    
    cfg.INPUT.IMAGE_TYPE = image_type
    cfg.INPUT.USE_UNDISTORTED = use_undistorted
    cfg.INPUT.NUM_VIEWS = num_views
    cfg.INPUT.MIN_VIEW_DISTANCE = min_view_distance
    cfg.INPUT.SIZE_DIVISIBILITY = 32
    cfg.INPUT.FORMAT = "RGB"
    
    # ===== TRAINING =====
    cfg.SOLVER.IMS_PER_BATCH = batch_size * num_gpus
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.DPT_LR = dpt_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.8),)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WARMUP_ITERS = min(1000, max_iter // 10)
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.AMP.ENABLED = True
    
    # ===== OUTPUT =====
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1000, max_iter // 10)
    cfg.TEST.EVAL_PERIOD = max(500, max_iter // 20)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    
    cfg.VERSION = 2
    
    return cfg


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_scannetpp(args):
    """Main training function for ScanNet++."""
    
    # Setup configuration
    cfg = setup_scannetpp_cfg(
        mapanything_checkpoint=args.mapanything_checkpoint,
        scannetpp_root=args.scannetpp_root,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        split_file=args.split_file,
        panoptic_dir=args.panoptic_dir,
        image_type=args.image_type,
        use_undistorted=args.use_undistorted,
        num_views=args.num_views,
        min_view_distance=args.min_view_distance,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dpt_lr=args.dpt_lr,
        max_iter=args.max_iter,
    )
    
    # Register dataset
    if args.scene_id:
        # Single scene mode
        train_name, val_name = register_scannetpp_single_scene(
            scene_id=args.scene_id,
            data_root=args.scannetpp_root,
            panoptic_dir=args.panoptic_dir,
            image_type=args.image_type,
            use_undistorted=args.use_undistorted,
            num_views_train=args.num_train_views,
            num_views_val=args.num_val_views,
            min_view_distance=args.min_view_distance,
        )
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (val_name,)
    else:
        # Multi-scene mode (using split file)
        if not args.split_file:
            raise ValueError("Either --scene-id or --split-file must be provided")
        
        register_scannetpp_panoptic(
            name="scannetpp_train",
            data_root=args.scannetpp_root,
            split_file=args.split_file,
            panoptic_dir=args.panoptic_dir or "",
            image_type=args.image_type,
            use_undistorted=args.use_undistorted,
        )
        cfg.DATASETS.TRAIN = ("scannetpp_train",)
        cfg.DATASETS.TEST = ()
    
    # Print configuration
    print("\n" + "="*80)
    print("SCANNET++ TRAINING CONFIGURATION")
    print("="*80)
    print(f"MapAnything checkpoint: {args.mapanything_checkpoint}")
    print(f"ScanNet++ root: {args.scannetpp_root}")
    print(f"Scene ID: {args.scene_id or 'Multi-scene mode'}")
    print(f"Image type: {args.image_type}")
    print(f"Undistorted: {args.use_undistorted}")
    print(f"Num views per sample: {args.num_views}")
    print(f"Min view distance: {args.min_view_distance}m")
    print(f"Batch size: {args.batch_size} x {args.num_gpus} GPUs")
    print(f"Max iterations: {args.max_iter}")
    print("="*80 + "\n")
    
    # Create trainer and train
    trainer = ScanNetPPTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train MapAnything + Mask2Former on ScanNet++",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--mapanything-checkpoint",
        type=str,
        required=True,
        help="Path to MapAnything pretrained checkpoint",
    )
    parser.add_argument(
        "--scannetpp-root",
        type=str,
        required=True,
        help="Root directory of ScanNet++ data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_scannetpp",
        help="Output directory for checkpoints and logs",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Single scene ID for training (for debugging/fine-tuning)",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to split file (e.g., nvs_sem_train.txt) for multi-scene training",
    )
    parser.add_argument(
        "--panoptic-dir",
        type=str,
        default=None,
        help="Directory with 2D panoptic annotations",
    )
    
    # Image configuration
    parser.add_argument(
        "--image-type",
        type=str,
        choices=['dslr', 'iphone'],
        default='dslr',
        help="Image type to use",
    )
    parser.add_argument(
        "--use-undistorted",
        action="store_true",
        default=True,
        help="Use undistorted pinhole images (recommended)",
    )
    parser.add_argument(
        "--no-undistorted",
        action="store_false",
        dest="use_undistorted",
        help="Use original fisheye images",
    )
    
    # View selection
    parser.add_argument(
        "--num-views",
        type=int,
        default=1,
        help="Number of views per training sample",
    )
    parser.add_argument(
        "--num-train-views",
        type=int,
        default=-1,
        help="Number of views for training (-1 = all available)",
    )
    parser.add_argument(
        "--num-val-views",
        type=int,
        default=-1,
        help="Number of views for validation (-1 = all available)",
    )
    parser.add_argument(
        "--min-view-distance",
        type=float,
        default=0.5,
        help="Minimum distance (m) between selected views",
    )
    
    # Training configuration
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Base learning rate",
    )
    parser.add_argument(
        "--dpt-lr",
        type=float,
        default=1e-5,
        help="Learning rate for DPT components",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    
    args = parser.parse_args()
    
    if args.num_gpus > 1:
        launch(
            train_scannetpp,
            args.num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(args,),
        )
    else:
        train_scannetpp(args)


if __name__ == "__main__":
    main()
