"""
Training script for Mask2Former with MapAnything Backbone + Panoptic DPT Head

THREE-COMPONENT PIPELINE:
1. Frozen MapAnything Backbone (DINOv2 + Multi-View Transformer)
2. Trainable Panoptic DPT Head (initialized from geometric DPT)
3. Trainable Mask2Former Head (query-based panoptic segmentation)

Key Features:
- MapAnything backbone frozen, outputs 768-dim tokens at H/14 resolution
- Panoptic DPT duplicates geometric DPT architecture (reassemble + fusion)
- DPT weights initialized from MapAnything's pretrained geometric head
- Multi-scale pyramid: res2(96), res3(192), res4(384), res5(768) channels
- Differential learning rates: DPT(1e-5), new projections + Mask2Former(1e-4)
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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

# Filter FutureWarnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# Ensure local mask2former is importable
sys.path.insert(0, os.getcwd())

from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# MapAnything imports
from mapanything.models import MapAnything


# ============================================================
# PANOPTIC DPT HEAD - DUPLICATED FROM GEOMETRIC DPT
# ============================================================

class ResidualConvUnit(nn.Module):
    """Residual convolution module from DPT architecture."""
    
    def __init__(self, features: int, activation=nn.ReLU(inplace=True), groups: int = 1):
        super().__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block from DPT architecture."""
    
    def __init__(self, features: int, activation=nn.ReLU(inplace=True), has_residual: bool = True, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.has_residual = has_residual
        
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, groups=groups)
        
        self.resConfUnit2 = ResidualConvUnit(features, activation, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, *xs, size=None):
        output = xs[0]
        
        if self.has_residual and len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        
        output = self.resConfUnit2(output)
        output = F.interpolate(output, size=size, mode="bilinear", align_corners=True) if size is not None else output
        output = self.out_conv(output)
        
        return output


class PanopticDPTHead(nn.Module):
    """
    Panoptic DPT Head - Duplicated from MapAnything's Geometric DPT.
    
    Takes 768-dim transformer tokens and produces multi-scale features for Mask2Former.
    
    Architecture:
    - Projects tokens from 4 transformer layers to intermediate channels
    - Applies reassemble (resize) operations to create pyramid
    - Fuses features through refinenet blocks
    - Projects to final channels: res2(96), res3(192), res4(384), res5(768)
    
    Args:
        dim_in: Input token dimension (768 for DINOv2)
        patch_size: Patch size of vision transformer (14 for DINOv2)
        features: Internal DPT feature dimension (256)
        out_channels: Intermediate projection channels [256, 512, 1024, 1024]
        output_channels: Final output channels for Mask2Former [96, 192, 384, 768]
        intermediate_layer_idx: Which transformer layers to use [4, 11, 17, 23]
    """
    
    def __init__(
        self,
        dim_in: int = 768,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        output_channels: List[int] = [96, 192, 384, 768],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.features = features
        
        self.norm = nn.LayerNorm(dim_in)
        
        # Projection layers (reassemble stage 1)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])
        
        # Resize layers (reassemble stage 2)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),  # 4x upsample
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),  # 2x upsample
            nn.Identity(),  # same size
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),  # 2x downsample
        ])
        
        # Scratch layers (channel adaptation before fusion)
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Fusion blocks (refinenet)
        self.scratch.refinenet1 = FeatureFusionBlock(features, has_residual=True)
        self.scratch.refinenet2 = FeatureFusionBlock(features, has_residual=True)
        self.scratch.refinenet3 = FeatureFusionBlock(features, has_residual=True)
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        
        # Output projections to Mask2Former expected channels
        # These are NEW (not from geometric DPT) - will be randomly initialized
        self.output_projections = nn.ModuleDict({
            'res2': nn.Conv2d(features, output_channels[0], kernel_size=1),  # 256 -> 96
            'res3': nn.Conv2d(features, output_channels[1], kernel_size=1),  # 256 -> 192
            'res4': nn.Conv2d(features, output_channels[2], kernel_size=1),  # 256 -> 384
            'res5': nn.Conv2d(features, output_channels[3], kernel_size=1),  # 256 -> 768
        })
        
        self._output_channels = {
            'res2': output_channels[0],
            'res3': output_channels[1],
            'res4': output_channels[2],
            'res5': output_channels[3],
        }
        
        print(f"\nPanopticDPTHead initialized:")
        print(f"  Input: {dim_in}-dim tokens from layers {intermediate_layer_idx}")
        print(f"  Internal features: {features}")
        print(f"  Output channels: res2({output_channels[0]}), res3({output_channels[1]}), res4({output_channels[2]}), res5({output_channels[3]})")
    
    def forward(self, aggregated_tokens_list: List[torch.Tensor], H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Panoptic DPT head.
        
        Args:
            aggregated_tokens_list: List of transformer layer outputs [B, N_tokens, 768]
            H, W: Original image dimensions
        
        Returns:
            Dictionary with res2, res3, res4, res5 features
        """
        B = aggregated_tokens_list[0].shape[0]
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Extract and reshape features from selected layers
        layer_features = []
        for idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]  # [B, N_tokens, 768]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)  # [B, 768, H/14, W/14]
            x = self.projects[idx](x)  # Project to intermediate channels
            x = self.resize_layers[idx](x)  # Resize for pyramid
            layer_features.append(x)
        
        # Apply scratch layers
        layer_1_rn = self.scratch.layer1_rn(layer_features[0])
        layer_2_rn = self.scratch.layer2_rn(layer_features[1])
        layer_3_rn = self.scratch.layer3_rn(layer_features[2])
        layer_4_rn = self.scratch.layer4_rn(layer_features[3])
        
        # Fusion (bottom-up)
        out4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out3 = self.scratch.refinenet3(out4, layer_3_rn, size=layer_2_rn.shape[2:])
        out2 = self.scratch.refinenet2(out3, layer_2_rn, size=layer_1_rn.shape[2:])
        out1 = self.scratch.refinenet1(out2, layer_1_rn)
        
        # Store intermediate outputs at different scales
        # These correspond to the pyramid levels Mask2Former expects
        outputs = {
            'res5': self.output_projections['res5'](out4),  # Smallest (stride 32)
            'res4': self.output_projections['res4'](out3),  # Medium (stride 16)
            'res3': self.output_projections['res3'](out2),  # Medium (stride 8)
            'res2': self.output_projections['res2'](out1),  # Largest (stride 4)
        }
        
        return outputs
    
    def output_channels(self) -> Dict[str, int]:
        return self._output_channels


# ============================================================
# FROZEN MAPANYTHING BACKBONE (Outputs Transformer Tokens)
# ============================================================

class FrozenMapAnythingBackbone(Backbone):
    """
    Detectron2-compatible wrapper for frozen MapAnything backbone.
    
    Outputs RAW transformer tokens (not processed through DPT).
    The Panoptic DPT head will process these tokens separately.
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        # Load pre-trained MapAnything model
        mapanything_path = cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH
        print(f"\nLoading frozen MapAnything from: {mapanything_path}")
        
        if not os.path.exists(mapanything_path):
            raise FileNotFoundError(f"MapAnything checkpoint not found: {mapanything_path}")

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
        
        # Storage for aggregated tokens from all transformer layers
        self.aggregated_tokens_list = []
        
        # We don't define output features here - the PanopticDPTHead will do that
        # This backbone just provides tokens
        self.token_dim = 768  # DINOv2 output dimension
        self.patch_size = 14  # DINOv2 patch size
    
    def forward(self, x):
        """
        Forward pass - extract transformer tokens.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Dict containing transformer tokens and image dimensions
        """
        batch_size, _, orig_h, orig_w = x.shape
        
        # DINOv2 requires dimensions divisible by patch_size (14)
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Prepare views for MapAnything (expects list of images)
        views = []
        for i in range(batch_size):
            img_np = x[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img_np = (img_np * 255).astype('uint8')
            views.append(Image.fromarray(img_np))
        
        # Extract features from MapAnything backbone
        with torch.no_grad():
            # Get transformer outputs from all layers
            output = self.mapanything(views, return_all_layers=True)
            
            # aggregated_tokens_list contains outputs from all transformer layers
            # Each element: [B, N_tokens, 768]
            self.aggregated_tokens_list = output.get('aggregated_tokens_list', [])
        
        # Return tokens and dimensions (PanopticDPTHead will process these)
        return {
            'tokens': self.aggregated_tokens_list,
            'H': orig_h,
            'W': orig_w,
        }
    
    def output_shape(self):
        # This backbone doesn't directly output feature maps
        # The PanopticDPTHead will define the actual output shapes
        return {}
    
    @property
    def size_divisibility(self):
        return self.patch_size


# ============================================================
# COMBINED BACKBONE + PANOPTIC DPT
# ============================================================

class MapAnythingWithPanopticDPT(Backbone):
    """
    Combined module: Frozen MapAnything + Trainable Panoptic DPT.
    
    This wraps both components to present a unified interface to Detectron2.
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        # Component 1: Frozen MapAnything Backbone
        self.backbone = FrozenMapAnythingBackbone(cfg, input_shape)
        
        # Component 2: Trainable Panoptic DPT Head
        self.panoptic_dpt = PanopticDPTHead(
            dim_in=768,
            patch_size=14,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            output_channels=[96, 192, 384, 768],  # Mask2Former expected channels
            intermediate_layer_idx=[4, 11, 17, 23],
        )
        
        # Initialize Panoptic DPT from frozen MapAnything's geometric DPT
        self._initialize_panoptic_dpt_from_geometric()
        
        self._out_feature_strides = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        
        self._out_features = ['res2', 'res3', 'res4', 'res5']
    
    def _initialize_panoptic_dpt_from_geometric(self):
        """
        Initialize Panoptic DPT weights from MapAnything's geometric DPT head.
        
        This copies:
        - norm layer
        - projects (token projection layers)
        - resize_layers (upsampling/downsampling)
        - scratch layers (layer*_rn convolutions)
        - refinenet fusion blocks
        
        Does NOT copy:
        - output_projections (these are new for panoptic task)
        """
        print("\n" + "="*60)
        print("INITIALIZING PANOPTIC DPT FROM GEOMETRIC DPT")
        print("="*60)
        
        if not hasattr(self.backbone.mapanything, 'dpt_feature_head'):
            print("WARNING: MapAnything model doesn't have dpt_feature_head")
            print("Panoptic DPT will use random initialization")
            return
        
        geometric_dpt = self.backbone.mapanything.dpt_feature_head
        panoptic_dpt = self.panoptic_dpt
        
        # Helper function to copy weights
        def copy_weights(src, dst, name):
            try:
                if hasattr(src, 'weight') and hasattr(dst, 'weight'):
                    dst.weight.data.copy_(src.weight.data)
                    print(f"  ✓ Copied {name}.weight")
                if hasattr(src, 'bias') and hasattr(dst, 'bias') and src.bias is not None:
                    dst.bias.data.copy_(src.bias.data)
                    print(f"  ✓ Copied {name}.bias")
            except Exception as e:
                print(f"  ✗ Failed to copy {name}: {e}")
        
        # Copy norm layer
        copy_weights(geometric_dpt.norm, panoptic_dpt.norm, "norm")
        
        # Copy projects (4 projection layers)
        for i in range(4):
            copy_weights(geometric_dpt.projects[i], panoptic_dpt.projects[i], f"projects[{i}]")
        
        # Copy resize layers
        for i in range(4):
            if not isinstance(geometric_dpt.resize_layers[i], nn.Identity):
                copy_weights(geometric_dpt.resize_layers[i], panoptic_dpt.resize_layers[i], f"resize_layers[{i}]")
        
        # Copy scratch layers
        for layer_name in ['layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn']:
            copy_weights(
                getattr(geometric_dpt.scratch, layer_name),
                getattr(panoptic_dpt.scratch, layer_name),
                f"scratch.{layer_name}"
            )
        
        # Copy refinenet blocks (these are complex, copy recursively)
        for refinenet_name in ['refinenet1', 'refinenet2', 'refinenet3', 'refinenet4']:
            src_block = getattr(geometric_dpt.scratch, refinenet_name)
            dst_block = getattr(panoptic_dpt.scratch, refinenet_name)
            
            # Copy out_conv
            copy_weights(src_block.out_conv, dst_block.out_conv, f"scratch.{refinenet_name}.out_conv")
            
            # Copy resConfUnit1 (if exists)
            if hasattr(src_block, 'resConfUnit1') and src_block.has_residual:
                for conv_name in ['conv1', 'conv2']:
                    copy_weights(
                        getattr(src_block.resConfUnit1, conv_name),
                        getattr(dst_block.resConfUnit1, conv_name),
                        f"scratch.{refinenet_name}.resConfUnit1.{conv_name}"
                    )
            
            # Copy resConfUnit2
            for conv_name in ['conv1', 'conv2']:
                copy_weights(
                    getattr(src_block.resConfUnit2, conv_name),
                    getattr(dst_block.resConfUnit2, conv_name),
                    f"scratch.{refinenet_name}.resConfUnit2.{conv_name}"
                )
        
        print("\n✅ Panoptic DPT initialized from geometric DPT!")
        print("   Output projections (res2/3/4/5) are randomly initialized")
        print("="*60 + "\n")
    
    def forward(self, x):
        """Forward through both components."""
        # Get transformer tokens from frozen backbone
        backbone_output = self.backbone(x)
        tokens = backbone_output['tokens']
        H = backbone_output['H']
        W = backbone_output['W']
        
        # Process through trainable Panoptic DPT
        features = self.panoptic_dpt(tokens, H, W)
        
        return features
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.panoptic_dpt.output_channels()[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    @property
    def size_divisibility(self):
        return self.backbone.size_divisibility
    
    def train(self, mode=True):
        """Ensure backbone stays frozen."""
        super().train(mode)
        self.backbone.mapanything.eval()
        self.panoptic_dpt.train(mode)
        return self


# ============================================================
# REGISTER BACKBONE TO DETECTRON2
# ============================================================

from detectron2.modeling.backbone import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
def build_mapanything_with_panoptic_dpt_backbone(cfg, input_shape):
    return MapAnythingWithPanopticDPT(cfg, input_shape)


# ============================================================
# CUSTOM TRAINER WITH DIFFERENTIAL LEARNING RATES
# ============================================================

class NaNLossCheckHook(HookBase):
    """Hook to check for NaN losses during training."""
    def after_step(self):
        if self.trainer.storage.latest().get("total_loss") is None:
            return
        
        loss_value = self.trainer.storage.latest()["total_loss"][0]
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            raise RuntimeError(f"NaN/Inf loss detected at iteration {self.trainer.iter}")


class Mask2FormerPanopticTrainer(DefaultTrainer):
    """Custom trainer with differential learning rates for DPT components."""
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(NaNLossCheckHook())
        return hooks
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build optimizer with differential learning rates:
        - Panoptic DPT (copied from geometric): 1e-5
        - Output projections (new): 1e-4
        - Mask2Former head: 1e-4
        """
        # Separate parameters into groups
        dpt_params = []
        projection_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone.panoptic_dpt' in name:
                if 'output_projections' in name:
                    projection_params.append(param)
                else:
                    dpt_params.append(param)
            else:
                other_params.append(param)
        
        # Build parameter groups with differential LRs
        params = [
            {
                'params': dpt_params,
                'lr': cfg.SOLVER.DPT_LR,  # Lower LR for copied DPT weights
                'name': 'panoptic_dpt'
            },
            {
                'params': projection_params,
                'lr': cfg.SOLVER.BASE_LR,  # Higher LR for new projections
                'name': 'dpt_projections'
            },
            {
                'params': other_params,
                'lr': cfg.SOLVER.BASE_LR,  # Higher LR for Mask2Former
                'name': 'mask2former'
            }
        ]
        
        print("\n" + "="*60)
        print("OPTIMIZER PARAMETER GROUPS:")
        print(f"  Panoptic DPT (copied weights): {len(dpt_params)} params @ LR={cfg.SOLVER.DPT_LR}")
        print(f"  DPT Output Projections (new): {len(projection_params)} params @ LR={cfg.SOLVER.BASE_LR}")
        print(f"  Mask2Former Head: {len(other_params)} params @ LR={cfg.SOLVER.BASE_LR}")
        print("="*60 + "\n")
        
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        
        return optimizer
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOPanopticEvaluator(dataset_name, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        # Add data augmentations
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip(horizontal=True),
        ]
        
        mapper = None  # Use default mapper
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)


# ============================================================
# SETUP CONFIGURATION
# ============================================================

def setup_cfg(
    mapanything_checkpoint_path: str,
    coco_root: str,
    output_dir: str,
    num_gpus: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    dpt_lr: float = 1e-5,
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
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_with_panoptic_dpt_backbone"
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
    
    # ===== MASK FORMER DECODER CONFIGURATION =====
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
    
    # ===== TRAINING CONFIGURATION =====
    cfg.SOLVER.IMS_PER_BATCH = batch_size * num_gpus
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.DPT_LR = dpt_lr  # New: Differential LR for DPT
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (70000,)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Not used (we handle LRs manually)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.AMP.ENABLED = True
    
    # ===== EVALUATION CONFIGURATION =====
    cfg.TEST.EVAL_PERIOD = 5000
    
    # ===== OUTPUT CONFIGURATION =====
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.VERSION = 2
    
    print("\n" + "="*80)
    print("THREE-COMPONENT PIPELINE CONFIGURATION:")
    print("="*80)
    print("1. FROZEN MapAnything Backbone:")
    print(f"   - DINOv2 + Multi-View Transformer")
    print(f"   - Output: 768-dim tokens at H/14 resolution")
    print("")
    print("2. TRAINABLE Panoptic DPT Head:")
    print(f"   - Initialized from geometric DPT")
    print(f"   - Learning rate: {dpt_lr} (DPT) / {learning_rate} (projections)")
    print(f"   - Output channels: res2(96), res3(192), res4(384), res5(768)")
    print("")
    print("3. TRAINABLE Mask2Former Head:")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Queries: 100, Decoder layers: 9")
    print("="*80 + "\n")
    
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

def train_panoptic_dpt(
    mapanything_checkpoint: str,
    coco_root: str,
    output_dir: str = "./output_panoptic_dpt",
    num_gpus: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    dpt_lr: float = 1e-5,
    max_iter: int = 90000,
    resume: bool = False,
):
    print("="*80)
    print("MASK2FORMER WITH MAPANYTHING + PANOPTIC DPT - THREE COMPONENT PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  MapAnything checkpoint: {mapanything_checkpoint}")
    print(f"  COCO root: {coco_root}")
    print(f"  Output directory: {output_dir}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (Mask2Former + projections)")
    print(f"  DPT learning rate: {dpt_lr} (copied DPT weights)")
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
        dpt_lr=dpt_lr,
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
    train_panoptic_dpt(
        mapanything_checkpoint=args.map_anything_checkpoint,
        coco_root=args.coco_root,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dpt_lr=args.dpt_lr,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    # ===== PATH CONFIGURATION =====
    BASE_DIR = os.getcwd()
    MAPANYTHING_CHECKPOINT = os.path.join(BASE_DIR, "pretrained_models", "map_anything", "test")
    COCO_ROOT = os.path.join(BASE_DIR, "datasets", "coco")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_panoptic_dpt")
    
    # ===== HYPERPARAMETERS =====
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Detected GPU: {gpu_name}")
    
    if "A100" in gpu_name:
        BATCH_SIZE = 8
        print(f"Configuring for A100 (80GB): Batch size {BATCH_SIZE}")
    else:
        BATCH_SIZE = 4
        print(f"Configuring for V100 (16GB): Batch size {BATCH_SIZE}")

    LEARNING_RATE = 1e-4  # For Mask2Former and new projections
    DPT_LR = 1e-5  # Lower LR for copied DPT weights
    MAX_ITER = 90000
    
    # Check paths
    if not os.path.exists(MAPANYTHING_CHECKPOINT):
        print(f"WARNING: MapAnything checkpoint not found at {MAPANYTHING_CHECKPOINT}")
        
    if not os.path.exists(COCO_ROOT):
        print(f"WARNING: COCO dataset not found at {COCO_ROOT}")

    args = default_argument_parser().parse_args()
    args.coco_root = COCO_ROOT
    args.output_dir = OUTPUT_DIR
    args.batch_size = BATCH_SIZE
    args.learning_rate = LEARNING_RATE
    args.dpt_lr = DPT_LR
    args.max_iter = MAX_ITER
    args.map_anything_checkpoint = MAPANYTHING_CHECKPOINT
    
    print("Command Line Args:", args)

    # Launch multi-GPU training
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
