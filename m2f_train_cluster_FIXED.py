"""
Training script for Mask2Former Panoptic Segmentation Head ONLY (Cluster Version) - FIXED
Optimized for GPU Cluster (16GB VRAM)

CRITICAL FIXES APPLIED:
- ✅ Added proper data augmentations (flip, multi-scale resize)
- ✅ Increased CLASS_WEIGHT from 2.0 to 5.0 (balance classification vs mask loss)
- ✅ Fixed LR schedule: Single decay at 70k instead of 60k+80k
- ✅ Added warmup for stable training start
- ✅ Improved projection layers (deeper 3-layer adapters)
- ✅ Better hyperparameters for frozen backbone training
"""
import warnings
# Filter specifically by the content of the message
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Dict
from PIL import Image
import cv2
import copy
from detectron2.layers import ShapeSpec
from detectron2.config import CfgNode as CN
from detectron2.engine import launch, default_argument_parser
# Detectron2 imports
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.modeling import Backbone
import detectron2.utils.comm as comm

# Ensure local mask2former is importable
# Assuming script is run from the project root where 'mask2former' package exists
sys.path.insert(0, os.getcwd())

from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# MapAnything imports
from mapanything.models import MapAnything


# ============================================================
# NAN CHECK HOOK
# ============================================================

class NaNLossCheckHook(HookBase):
    """
    Hook to check for NaN losses during training and exit early if found.
    """
    def after_step(self):
        if self.trainer.storage.latest().get("total_loss") is None:
            return
        
        loss_value = self.trainer.storage.latest()["total_loss"][0]
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            raise FloatingPointError(f"Loss became {loss_value} at iteration {self.trainer.iter}. Exiting early.")


# ============================================================
# FROZEN MAPANYTHING BACKBONE (Detectron2-compatible) - IMPROVED
# ============================================================

class FrozenMapAnythingBackbone(Backbone):
    """
    Detectron2-compatible wrapper for frozen MapAnything backbone.
    Extracts multi-scale features from MapAnything's DPT refinenet layers.
    
    IMPROVED: Uses deeper 3-layer projection adapters instead of 1x1 convs
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        # Load pre-trained MapAnything model
        mapanything_path = cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH
        print(f"Loading frozen MapAnything from: {mapanything_path}")
        
        if not os.path.exists(mapanything_path):
             # Fallback for relative path if absolute path fails
             if os.path.exists(os.path.abspath(mapanything_path)):
                 mapanything_path = os.path.abspath(mapanything_path)
             else:
                 raise FileNotFoundError(f"MapAnything checkpoint not found at: {mapanything_path}")

        self.mapanything = MapAnything.from_pretrained(
            mapanything_path,
            local_files_only=True
        )
        
        # FREEZE all MapAnything parameters
        self.mapanything.eval()
        for param in self.mapanything.parameters():
            param.requires_grad = False
        
        print("MapAnything loaded and FROZEN")
        print(f"  Total frozen parameters: {sum(p.numel() for p in self.mapanything.parameters()):,}")
        
        # Storage for features
        self.features = {}
        self.hooks = []
        
        # Register hooks on DPT refinenet layers
        self._register_dpt_hooks()
        
        # Define output feature channels and strides for Detectron2
        self._out_feature_channels = {
            'res2': 256,   # refinenet4 - H/4
            'res3': 512,   # refinenet3 - H/8
            'res4': 1024,  # refinenet2 - H/16
            'res5': 2048,  # refinenet1 - H/32
        }
        
        self._out_feature_strides = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        
        # Create trainable projection layers (IMPROVED)
        self._create_projection_layers()
    
    def _create_projection_layers(self):
        """Create DEEPER trainable projection layers (3-layer adapters)"""
        self.projections = nn.ModuleDict()
        dpt_channels = 256
        
        for feat_name, target_channels in self._out_feature_channels.items():
            if target_channels != dpt_channels:
                # Use 3-layer adapter instead of simple 1x1 conv
                self.projections[feat_name] = nn.Sequential(
                    # Layer 1: 3x3 conv with same channels
                    nn.Conv2d(dpt_channels, dpt_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, dpt_channels),
                    nn.ReLU(inplace=True),
                    
                    # Layer 2: 3x3 conv to target channels
                    nn.Conv2d(dpt_channels, target_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, target_channels),
                    nn.ReLU(inplace=True),
                    
                    # Layer 3: 1x1 conv for final refinement
                    nn.Conv2d(target_channels, target_channels, kernel_size=1)
                )
            else:
                self.projections[feat_name] = nn.Identity()
        
        # Count trainable parameters
        total_params = sum(p.numel() for module in self.projections.values() 
                          if not isinstance(module, nn.Identity) 
                          for p in module.parameters())
        print(f"Created DEEP projection adapters with {total_params:,} trainable parameters")
    
    def _register_dpt_hooks(self):
        """Register hooks on MapAnything's DPT refinenet layers."""
        print("\nRegistering hooks on DPT refinenet layers...")
        
        mapping = {
            'refinenet4': 'res2',
            'refinenet3': 'res3',
            'refinenet2': 'res4',
            'refinenet1': 'res5'
        }
        
        def get_hook(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        try:
            if hasattr(self.mapanything, 'dpt_feature_head'):
                scratch = self.mapanything.dpt_feature_head.scratch
                
                for dpt_name, res_name in mapping.items():
                    if hasattr(scratch, dpt_name):
                        layer = getattr(scratch, dpt_name)
                        self.hooks.append(layer.register_forward_hook(get_hook(res_name)))
                        print(f"  Registered hook for {res_name} on {dpt_name}")
                
                if len(self.hooks) == 0:
                    print("  WARNING: No hooks were registered!")
                else:
                    print(f"  Successfully registered {len(self.hooks)} hooks")
            else:
                 print("WARNING: dpt_feature_head not found! Check MapAnything structure.")

        except AttributeError as e:
            print(f"Error accessing DPT layers: {e}. Check model structure.")
    
    def forward(self, x):
        """Forward pass compatible with Detectron2."""
        self.features = {}
        batch_size, _, orig_h, orig_w = x.shape
        
        # DINOv2 requires dimensions divisible by patch_size (14)
        patch_size = 14
        pad_h = (patch_size - orig_h % patch_size) % patch_size
        pad_w = (patch_size - orig_w % patch_size) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        views = []
        for i in range(batch_size):
            view = {
                'img': x[i:i+1],
                'data_norm_type': ['dinov2'],
                'instance': [f'view_{i}'],
                'idx': [i],
            }
            views.append(view)
        
        with torch.no_grad():
            try:
                _ = self.mapanything(views)
            except torch.cuda.OutOfMemoryError as e:
                # If OOM in dense head, that's okay - hooks already captured features
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: MapAnything forward failed: {e}")
        
        output_features = {}
        
        if not self.features:
            raise RuntimeError("No features were captured from MapAnything!")
        
        for feat_name in self._out_features:
            if feat_name not in self.features:
                continue
            
            feat = self.features[feat_name]
            
            if pad_h > 0 or pad_w > 0:
                stride = self._out_feature_strides[feat_name]
                target_h = orig_h // stride
                target_w = orig_w // stride
                feat = feat[:, :, :target_h, :target_w]
            
            feat = self.projections[feat_name](feat)
            output_features[feat_name] = feat
        
        if not output_features:
            raise RuntimeError("No output features produced!")
        
        return output_features
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def train(self, mode=True):
        super().train(mode)
        self.mapanything.eval()
        for proj in self.projections.values():
            if not isinstance(proj, nn.Identity):
                proj.train(mode)
        return self
    
    @property
    def size_divisibility(self):
        return 32


# ============================================================
# REGISTER MAPANYTHING BACKBONE TO DETECTRON2
# ============================================================

from detectron2.modeling.backbone import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
def build_mapanything_backbone(cfg, input_shape):
    return FrozenMapAnythingBackbone(cfg, input_shape)


# ============================================================
# CUSTOM TRAINER FOR PANOPTIC SEGMENTATION - WITH AUGMENTATIONS
# ============================================================

class Mask2FormerPanopticTrainer(DefaultTrainer):
    """
    Custom trainer for Mask2Former panoptic segmentation.
    FIXED: Now includes proper data augmentations!
    """
    
    def build_hooks(self):
        """Add custom hooks"""
        hooks = super().build_hooks()
        hooks.append(NaNLossCheckHook())
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOPanopticEvaluator(dataset_name, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
            MaskFormerPanopticDatasetMapper,
        )
        
        # CRITICAL FIX: Add proper data augmentations!
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        ]
        
        print("="*60)
        print("DATA AUGMENTATIONS ENABLED:")
        print("  - Multi-scale resize: [640-800] -> max 1333")
        print("  - Random horizontal flip: 50% probability")
        print("="*60)
        
        mapper = MaskFormerPanopticDatasetMapper(
            is_train=True,
            augmentations=augmentations,  # ← FIXED: No longer empty!
            image_format=cfg.INPUT.FORMAT,
            ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        )
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def resume_or_load(self, resume=True):
        super().resume_or_load(resume=resume)
        
        model = self.model
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
            
        if hasattr(model, "backbone") and isinstance(model.backbone, FrozenMapAnythingBackbone):
            print("Re-freezing MapAnything backbone parameters...")
            model.backbone.mapanything.eval()
            for param in model.backbone.mapanything.parameters():
                param.requires_grad = False
            
            for proj in model.backbone.projections.values():
                if not isinstance(proj, nn.Identity):
                    proj.train()
                    for param in proj.parameters():
                        param.requires_grad = True
            print("Backbone re-frozen, deep adapters trainable.")


# ============================================================
# SETUP CONFIGURATION - IMPROVED HYPERPARAMETERS
# ============================================================
def setup_cfg(
    mapanything_checkpoint_path: str,
    coco_root: str,
    output_dir: str,
    num_gpus: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    max_iter: int = 90000,
):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    
    # ===== MODEL ARCHITECTURE =====
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # ===== BACKBONE CONFIGURATION =====
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    cfg.MODEL.MAPANYTHING = CN()
    cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH = mapanything_checkpoint_path
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # ===== MASK2FORMER HEAD CONFIGURATION =====
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 133
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    
    # ===== PIXEL DECODER CONFIGURATION =====
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    
    # ===== MASK FORMER DECODER CONFIGURATION - IMPROVED LOSS WEIGHTS =====
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 5.0  # ← FIXED: Increased from 2.0 to 5.0
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
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    
    # ===== INPUT CONFIGURATION =====
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_panoptic"
    
    # ===== DATASET CONFIGURATION =====
    cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic",)
    cfg.DATASETS.TEST = ("coco_2017_val_panoptic",)
    
    # ===== TRAINING CONFIGURATION - IMPROVED LR SCHEDULE =====
    cfg.SOLVER.IMS_PER_BATCH = batch_size * num_gpus
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    
    # FIXED LR schedule: Single decay at 70k instead of 60k+80k
    cfg.SOLVER.STEPS = (70000,)  # ← FIXED: Single step at 70k
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    # Add warmup for stable training start
    cfg.SOLVER.WARMUP_ITERS = 1000  # ← NEW: 1000 iteration warmup
    cfg.SOLVER.WARMUP_FACTOR = 0.001  # ← NEW: Start from LR/1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    cfg.SOLVER.AMP.ENABLED = True
    
    # ===== EVALUATION CONFIGURATION =====
    cfg.TEST.EVAL_PERIOD = 5000  # ← More frequent eval to monitor progress
    
    # ===== OUTPUT CONFIGURATION =====
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000  # ← More frequent checkpoints
    
    cfg.DATALOADER.NUM_WORKERS = 16  # Increased for cluster
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.VERSION = 2
    
    print("\n" + "="*60)
    print("IMPROVED HYPERPARAMETERS:")
    print(f"  CLASS_WEIGHT: 5.0 (was 2.0) - Better classification")
    print(f"  LR Schedule: Decay at 70k (was 60k+80k)")
    print(f"  Warmup: 1000 iters from LR={learning_rate*0.001:.2e}")
    print(f"  Eval Period: 5000 (was 10000)")
    print(f"  Checkpoint Period: 5000 (was 10000)")
    print("="*60 + "\n")
    
    return cfg


# ============================================================
# REGISTER COCO PANOPTIC DATASET
# ============================================================

def register_coco_panoptic(coco_root: str):
    """Register COCO panoptic dataset with Detectron2."""
    from detectron2.data.datasets import register_coco_panoptic
    
    coco_root = os.path.abspath(coco_root)
    
    train_images = os.path.join(coco_root, "train2017")
    val_images = os.path.join(coco_root, "val2017")
    
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Training images not found: {train_images}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images not found: {val_images}")
    
    print(f"Found training images: {train_images}")
    print(f"Found validation images: {val_images}")
    
    # Register training set
    if "coco_2017_train_panoptic" not in DatasetCatalog:
        register_coco_panoptic(
            name="coco_2017_train_panoptic",
            metadata={},
            image_root=train_images,
            panoptic_root=os.path.join(coco_root, "panoptic_train2017"),
            panoptic_json=os.path.join(coco_root, "annotations/panoptic_train2017.json"),
            instances_json=os.path.join(coco_root, "annotations/instances_train2017.json"),
        )
    
    # Register validation set
    if "coco_2017_val_panoptic" not in DatasetCatalog:
        register_coco_panoptic(
            name="coco_2017_val_panoptic",
            metadata={},
            image_root=val_images,
            panoptic_root=os.path.join(coco_root, "panoptic_val2017"),
            panoptic_json=os.path.join(coco_root, "annotations/panoptic_val2017.json"),
            instances_json=os.path.join(coco_root, "annotations/instances_val2017.json"),
        )
    
    print("COCO panoptic dataset registered")


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_panoptic_segmentation_head(
    mapanything_checkpoint: str,
    coco_root: str,
    output_dir: str = "./output_panoptic",
    num_gpus: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    max_iter: int = 90000,
    resume: bool = False,
):
    print("="*80)
    print("MASK2FORMER PANOPTIC SEGMENTATION HEAD TRAINING - FIXED VERSION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  MapAnything checkpoint: {mapanything_checkpoint}")
    print(f"  COCO root: {coco_root}")
    print(f"  Output directory: {output_dir}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max iterations: {max_iter}")
    
    # Register COCO panoptic dataset
    register_coco_panoptic(coco_root)
    
    # Setup config
    cfg = setup_cfg(
        mapanything_checkpoint_path=mapanything_checkpoint,
        coco_root=coco_root,
        output_dir=output_dir,
        num_gpus=num_gpus,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    
    # Create trainer
    print("\nStarting training...")
    trainer = Mask2FormerPanopticTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    
    # Train!
    trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(args):
    """Wrapper function for multi-GPU launch"""
    train_panoptic_segmentation_head(
        mapanything_checkpoint=args.map_anything_checkpoint,
        coco_root=args.coco_root,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    
    # ===== RELATIVE PATH CONFIGURATION =====
    # Assumes script is run from project root
    BASE_DIR = os.getcwd()
    
    # 1. MapAnything Checkpoint
    # Expected: ./pretrained_models/map_anything/test
    MAPANYTHING_CHECKPOINT = os.path.join(BASE_DIR, "pretrained_models", "map_anything", "test")
    
    # 2. COCO Dataset
    # Expected: ./datasets/coco
    COCO_ROOT = os.path.join(BASE_DIR, "datasets", "coco")
    
    # 3. Output Directory
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_cluster_FIXED")  # New output dir
    
    # ===== CLUSTER HYPERPARAMETERS =====
    # Optimized for 16GB VRAM (Nvidia V100)
   # NUM_GPUS = torch.cuda.device_count()
    
    # Batch size logic based on GPU type
    # V100 (16GB): Batch size 2 per GPU
    # A100 (80GB): Batch size 8-10 per GPU
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Detected GPU: {gpu_name}")
    
    if "A100" in gpu_name:
        BATCH_SIZE = 8  # Aggressive batch size for 80GB A100
        print(f"Configuring for A100 (80GB): Batch size {BATCH_SIZE}")
    else:
        BATCH_SIZE = 4  # Conservative batch size for V100 (16GB)
        print(f"Configuring for V100 (16GB): Batch size {BATCH_SIZE}")

    LEARNING_RATE = 1e-4
    MAX_ITER = 90000
    
    # Check if paths exist
    if not os.path.exists(MAPANYTHING_CHECKPOINT):
        print(f"WARNING: MapAnything checkpoint not found at {MAPANYTHING_CHECKPOINT}")
        print("  Please ensure 'pretrained_models/map_anything/test' exists.")
        
    if not os.path.exists(COCO_ROOT):
        print(f"WARNING: COCO dataset not found at {COCO_ROOT}")
        print("  Please ensure 'datasets/coco' exists.")

    args = default_argument_parser().parse_args()
    args.coco_root = COCO_ROOT
    args.output_dir = OUTPUT_DIR
    args.batch_size = BATCH_SIZE
    args.learning_rate = LEARNING_RATE
    args.max_iter = MAX_ITER
    args.map_anything_checkpoint = MAPANYTHING_CHECKPOINT
    print("Command Line Args:", args)

    # Launch multi-GPU training
    launch(
        main,                    # Function to run
        args.num_gpus,          # Number of GPUs
        num_machines=1,         # Number of machines (1 for single node)
        machine_rank=0,         # This machine's rank (0 for single node)
        dist_url="auto",        # Automatically find a free port
        args=(args,),           # Pass args to main()
    )
