"""
Training script for Mask2Former with MapAnything Backbone + Panoptic DPT Head

THREE-COMPONENT PIPELINE:
1. Frozen MapAnything Backbone (DINOv2 + Multi-View Transformer)
2. Trainable Panoptic DPT Head (initialized from geometric DPT)
3. Trainable Mask2Former Head (query-based panoptic segmentation)

Key Features:
- MapAnything backbone frozen, outputs 768-dim transformer features
- Panoptic DPT duplicates geometric DPT architecture (reassemble + fusion)
- DPT weights initialized from MapAnything's pretrained geometric head
- Multi-scale pyramid: res2(96), res3(192), res4(384), res5(768) channels
- Differential learning rates: DPT(1e-5), new projections + Mask2Former(1e-4)

Architecture:
    Input Image [B, 3, H, W]
        ↓
    ┌─────────────────────────────────────────┐
    │  FROZEN: MapAnything Backbone           │
    │  - DINOv2 Encoder                       │
    │  - Multi-View Transformer (N=1 for COCO)│
    │  Output: List of 4 layer features       │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │  TRAINABLE: Panoptic DPT Head           │
    │  - Initialized from Geometric DPT       │
    │  - Reassemble + Fusion modules          │
    │  - New output projections               │
    │  Output: {res2, res3, res4, res5}       │
    └─────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────┐
    │  TRAINABLE: Mask2Former Head            │
    │  - MSDeformAttn Pixel Decoder           │
    │  - Transformer Decoder (masked attn)    │
    │  - Classification + Mask heads          │
    │  Output: Panoptic Predictions           │
    └─────────────────────────────────────────┘
"""

import os
import sys
import copy
import warnings
from typing import Dict, List, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
import detectron2.utils.comm as comm

# Ensure local mask2former is importable
sys.path.insert(0, os.getcwd())

from mask2former import add_maskformer2_config
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

# MapAnything imports
from mapanything.models import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput


# ============================================================
# NAN CHECK HOOK
# ============================================================

class NaNLossCheckHook(HookBase):
    """Hook to check for NaN losses during training and exit early if found."""
    def after_step(self):
        if self.trainer.storage.latest().get("total_loss") is None:
            return
        
        loss_value = self.trainer.storage.latest()["total_loss"][0]
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            raise FloatingPointError(f"Loss became {loss_value} at iteration {self.trainer.iter}. Exiting early.")


# ============================================================
# PANOPTIC DPT HEAD COMPONENTS
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
        
        if self.has_residual and len(xs) >= 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        
        output = self.resConfUnit2(output)
        
        if size is not None:
            output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
        
        output = self.out_conv(output)
        
        return output


class PanopticDPTHead(nn.Module):
    """
    Panoptic DPT Head - Duplicated architecture from MapAnything's Geometric DPT.
    
    Takes transformer features and produces multi-scale features for Mask2Former.
    
    Architecture:
    - Projects features from 4 transformer layers to intermediate channels
    - Each layer can have different input dimensions (encoder vs info_sharing)
    - Applies reassemble (resize) operations to create pyramid
    - Fuses features through refinenet blocks
    - Projects to final channels: res2(96), res3(192), res4(384), res5(768)
    
    Args:
        input_dims: List of input dimensions for each layer [768, 1024, 1024, 1024]
        patch_size: Patch size of vision transformer (14 for DINOv2)
        features: Internal DPT feature dimension (256)
        out_channels: Intermediate projection channels [256, 512, 1024, 1024]
        output_channels: Final output channels for Mask2Former [96, 192, 384, 768]
    """
    
    def __init__(
        self,
        input_dims: List[int] = None,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = None,
        output_channels: List[int] = None,
    ):
        super().__init__()
        
        if input_dims is None:
            input_dims = [768, 1024, 1024, 1024]  # encoder + 3x info_sharing
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        if output_channels is None:
            output_channels = [96, 192, 384, 768]
            
        self.patch_size = patch_size
        self.features = features
        self.input_dims = input_dims
        
        # Per-layer norm layers (each layer can have different input dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in input_dims
        ])
        
        # Projection layers (reassemble stage 1) - project from token dim to intermediate channels
        # Each projection handles the corresponding layer's input dimension
        self.projects = nn.ModuleList([
            nn.Conv2d(input_dims[i], out_channels[i], kernel_size=1, stride=1, padding=0)
            for i in range(4)
        ])
        
        # Resize layers (reassemble stage 2) - create multi-scale pyramid
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
        
        # Fusion blocks (refinenet) - bottom-up feature fusion
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
        
        self._output_strides = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }
        
        print(f"\nPanopticDPTHead initialized:")
        print(f"  Input dims: {input_dims}")
        print(f"  Internal features: {features}")
        print(f"  Output channels: res2({output_channels[0]}), res3({output_channels[1]}), res4({output_channels[2]}), res5({output_channels[3]})")
    
    def forward(self, layer_features_list: List[torch.Tensor], H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Panoptic DPT head.
        
        Args:
            layer_features_list: List of 4 transformer layer features, each [B*V, N, C] or [B*V, C, patch_h, patch_w]
            H, W: Original image dimensions
        
        Returns:
            Dictionary with res2, res3, res4, res5 features at appropriate strides
        """
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        expected_num_tokens = patch_h * patch_w
        
        # Process features through reassemble stages
        processed_features = []
        for idx, feat in enumerate(layer_features_list):
            expected_dim = self.input_dims[idx]
            
            # Handle different input formats
            if feat.dim() == 3:
                # [B, N, C] format - reshape to spatial
                B, N, C = feat.shape
                
                # Check if we need to strip register/CLS tokens
                if N > expected_num_tokens:
                    # Assume extra tokens are at the beginning (register tokens)
                    num_extra = N - expected_num_tokens
                    feat = feat[:, num_extra:, :]  # Take only patch tokens
                    N = expected_num_tokens
                elif N < expected_num_tokens:
                    # This shouldn't happen - print warning
                    print(f"WARNING: Layer {idx} has {N} tokens, expected {expected_num_tokens}")
                
                x = self.norms[idx](feat)
                x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)  # [B, C, H/14, W/14]
            else:
                # [B, C, H, W] format - already spatial
                B, C, fh, fw = feat.shape
                # Flatten, normalize, reshape back
                x = feat.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N, C]
                x = self.norms[idx](x)
                x = x.permute(0, 2, 1).reshape(B, C, fh, fw)
            
            # Project to intermediate channels
            x = self.projects[idx](x)
            # Resize for pyramid
            x = self.resize_layers[idx](x)
            processed_features.append(x)
        
        # Apply scratch layers (channel adaptation)
        layer_1_rn = self.scratch.layer1_rn(processed_features[0])
        layer_2_rn = self.scratch.layer2_rn(processed_features[1])
        layer_3_rn = self.scratch.layer3_rn(processed_features[2])
        layer_4_rn = self.scratch.layer4_rn(processed_features[3])
        
        # Fusion (bottom-up) - refinenet4 is deepest (smallest spatial size)
        # refinenet4: layer_4_rn only (no residual)
        out4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        
        # refinenet3: out4 + layer_3_rn
        out3 = self.scratch.refinenet3(out4, layer_3_rn, size=layer_2_rn.shape[2:])
        
        # refinenet2: out3 + layer_2_rn
        out2 = self.scratch.refinenet2(out3, layer_2_rn, size=layer_1_rn.shape[2:])
        
        # refinenet1: out2 + layer_1_rn (largest resolution)
        out1 = self.scratch.refinenet1(out2, layer_1_rn)
        
        # Apply output projections and resize to expected strides
        # out1 is at patch resolution (H/14), need to get to H/4 for res2
        outputs = {}
        
        # res5: stride 32 (smallest)
        res5 = self.output_projections['res5'](out4)
        target_h5, target_w5 = H // 32, W // 32
        outputs['res5'] = F.interpolate(res5, size=(target_h5, target_w5), mode='bilinear', align_corners=True)
        
        # res4: stride 16
        res4 = self.output_projections['res4'](out3)
        target_h4, target_w4 = H // 16, W // 16
        outputs['res4'] = F.interpolate(res4, size=(target_h4, target_w4), mode='bilinear', align_corners=True)
        
        # res3: stride 8
        res3 = self.output_projections['res3'](out2)
        target_h3, target_w3 = H // 8, W // 8
        outputs['res3'] = F.interpolate(res3, size=(target_h3, target_w3), mode='bilinear', align_corners=True)
        
        # res2: stride 4 (largest)
        res2 = self.output_projections['res2'](out1)
        target_h2, target_w2 = H // 4, W // 4
        outputs['res2'] = F.interpolate(res2, size=(target_h2, target_w2), mode='bilinear', align_corners=True)
        
        return outputs
    
    def output_channels(self) -> Dict[str, int]:
        return self._output_channels
    
    def output_strides(self) -> Dict[str, int]:
        return self._output_strides


# ============================================================
# MAPANYTHING WITH PANOPTIC DPT BACKBONE
# ============================================================

class MapAnythingWithPanopticDPT(Backbone):
    """
    Combined module: Frozen MapAnything + Trainable Panoptic DPT.
    
    This wrapper:
    1. Loads MapAnything and freezes all parameters
    2. Creates a new PanopticDPT head with same architecture as geometric DPT
    3. Copies weights from geometric DPT to panoptic DPT
    4. Registers hooks to capture intermediate transformer features
    5. Presents a unified Detectron2 Backbone interface
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        # Load pre-trained MapAnything model
        mapanything_path = cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH
        print(f"\n{'='*60}")
        print("LOADING MAPANYTHING WITH PANOPTIC DPT PIPELINE")
        print(f"{'='*60}")
        print(f"Loading MapAnything from: {mapanything_path}")
        
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
        
        frozen_params = sum(p.numel() for p in self.mapanything.parameters())
        print(f"MapAnything loaded and FROZEN ({frozen_params:,} parameters)")
        
        # Get key properties from MapAnything
        self.patch_size = getattr(self.mapanything.encoder, 'patch_size', 14)
        self.encoder_dim = getattr(self.mapanything.encoder, 'enc_embed_dim', 768)
        self.info_sharing_dim = getattr(self.mapanything.info_sharing, 'dim', 1024)
        
        # Check if MapAnything uses encoder features for DPT
        self.use_encoder_features_for_dpt = getattr(self.mapanything, 'use_encoder_features_for_dpt', True)
        
        # Determine input dimensions for each layer
        # Layer 0: encoder features (768), Layers 1-3: info_sharing features (1024)
        if self.use_encoder_features_for_dpt:
            input_dims = [self.encoder_dim] + [self.info_sharing_dim] * 3
        else:
            input_dims = [self.info_sharing_dim] * 4
        
        print(f"Feature dimensions per layer: {input_dims}")
        
        # Create Panoptic DPT Head with same architecture
        self.panoptic_dpt = PanopticDPTHead(
            input_dims=input_dims,
            patch_size=self.patch_size,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            output_channels=[96, 192, 384, 768],  # Mask2Former expected channels
        )
        
        # Initialize Panoptic DPT from geometric DPT weights
        self._initialize_panoptic_dpt_from_geometric()
        
        # Define output specs for Detectron2
        self._out_feature_strides = self.panoptic_dpt.output_strides()
        self._out_feature_channels = self.panoptic_dpt.output_channels()
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        
        trainable_params = sum(p.numel() for p in self.panoptic_dpt.parameters() if p.requires_grad)
        print(f"Panoptic DPT Head: {trainable_params:,} trainable parameters")
        print(f"{'='*60}\n")
    
    def _initialize_panoptic_dpt_from_geometric(self):
        """
        Initialize Panoptic DPT weights from MapAnything's geometric DPT head.
        
        Copies:
        - norm layer
        - projects (token projection layers)
        - resize_layers (upsampling/downsampling)
        - scratch layers (layer*_rn convolutions)
        - refinenet fusion blocks
        
        Does NOT copy:
        - output_projections (these are new for panoptic task)
        - norms and projects if dimensions don't match (different architecture)
        """
        print("\nInitializing Panoptic DPT from Geometric DPT...")
        
        # Find the geometric DPT head in MapAnything
        geometric_dpt = None
        if hasattr(self.mapanything, 'dpt_feature_head'):
            geometric_dpt = self.mapanything.dpt_feature_head
        elif hasattr(self.mapanything, 'dense_head') and len(self.mapanything.dense_head) > 0:
            geometric_dpt = self.mapanything.dense_head[0]  # First module is DPTFeature
        
        if geometric_dpt is None:
            print("  WARNING: Could not find geometric DPT head in MapAnything")
            print("  Panoptic DPT will use random initialization")
            return
        
        panoptic_dpt = self.panoptic_dpt
        copied_count = 0
        skipped_count = 0
        
        def copy_weights(src, dst, name):
            nonlocal copied_count, skipped_count
            try:
                if hasattr(src, 'weight') and hasattr(dst, 'weight'):
                    if src.weight.shape == dst.weight.shape:
                        dst.weight.data.copy_(src.weight.data)
                        copied_count += 1
                    else:
                        print(f"  ⚠ Shape mismatch for {name}.weight: {src.weight.shape} vs {dst.weight.shape} (skipped)")
                        skipped_count += 1
                        return False
                if hasattr(src, 'bias') and hasattr(dst, 'bias') and src.bias is not None and dst.bias is not None:
                    if src.bias.shape == dst.bias.shape:
                        dst.bias.data.copy_(src.bias.data)
                return True
            except Exception as e:
                print(f"  ✗ Failed to copy {name}: {e}")
                skipped_count += 1
                return False
        
        # NOTE: We do NOT copy norm layers - uniception DPTFeature may have different norm structure
        # Our PanopticDPTHead uses per-layer norms that are fresh initialized
        print("  ℹ Norm layers: Using fresh initialization (different architecture)")
        
        # NOTE: We do NOT copy projects - they have different input dimensions
        # uniception may use input_feature_dims which differs from our per-layer setup
        print("  ℹ Project layers: Using fresh initialization (input dims may differ)")
        
        # Copy resize layers (these should match - same out_channels)
        if hasattr(geometric_dpt, 'resize_layers'):
            for i in range(min(len(geometric_dpt.resize_layers), len(panoptic_dpt.resize_layers))):
                if not isinstance(geometric_dpt.resize_layers[i], nn.Identity):
                    copy_weights(geometric_dpt.resize_layers[i], panoptic_dpt.resize_layers[i], f"resize_layers[{i}]")
        
        # Copy scratch layers
        if hasattr(geometric_dpt, 'scratch'):
            for layer_name in ['layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn']:
                if hasattr(geometric_dpt.scratch, layer_name) and hasattr(panoptic_dpt.scratch, layer_name):
                    copy_weights(
                        getattr(geometric_dpt.scratch, layer_name),
                        getattr(panoptic_dpt.scratch, layer_name),
                        f"scratch.{layer_name}"
                    )
            
            # Copy refinenet blocks
            for refinenet_name in ['refinenet1', 'refinenet2', 'refinenet3', 'refinenet4']:
                if hasattr(geometric_dpt.scratch, refinenet_name) and hasattr(panoptic_dpt.scratch, refinenet_name):
                    src_block = getattr(geometric_dpt.scratch, refinenet_name)
                    dst_block = getattr(panoptic_dpt.scratch, refinenet_name)
                    
                    # Copy out_conv
                    if hasattr(src_block, 'out_conv'):
                        copy_weights(src_block.out_conv, dst_block.out_conv, f"scratch.{refinenet_name}.out_conv")
                    
                    # Copy resConfUnit1 (if exists)
                    if hasattr(src_block, 'resConfUnit1') and hasattr(dst_block, 'resConfUnit1'):
                        for conv_name in ['conv1', 'conv2']:
                            if hasattr(src_block.resConfUnit1, conv_name) and hasattr(dst_block.resConfUnit1, conv_name):
                                copy_weights(
                                    getattr(src_block.resConfUnit1, conv_name),
                                    getattr(dst_block.resConfUnit1, conv_name),
                                    f"scratch.{refinenet_name}.resConfUnit1.{conv_name}"
                                )
                    
                    # Copy resConfUnit2
                    if hasattr(src_block, 'resConfUnit2') and hasattr(dst_block, 'resConfUnit2'):
                        for conv_name in ['conv1', 'conv2']:
                            if hasattr(src_block.resConfUnit2, conv_name) and hasattr(dst_block.resConfUnit2, conv_name):
                                copy_weights(
                                    getattr(src_block.resConfUnit2, conv_name),
                                    getattr(dst_block.resConfUnit2, conv_name),
                                    f"scratch.{refinenet_name}.resConfUnit2.{conv_name}"
                                )
        
        print(f"  ✓ Copied {copied_count} weight tensors from geometric DPT (skipped {skipped_count} due to shape mismatch)")
        print("  ✓ Norms, projects, and output projections initialized randomly")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MapAnything backbone and Panoptic DPT.
        
        Directly calls MapAnything's internal methods to get the exact same features
        that the geometric DPT head uses.
        
        Args:
            x: Input images [B, 3, H, W] - already normalized by Detectron2
        
        Returns:
            Dictionary with res2, res3, res4, res5 features
        """
        batch_size, _, orig_h, orig_w = x.shape
        
        # Pad to be divisible by patch_size
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        padded_h, padded_w = x.shape[2], x.shape[3]
        
        # ============================================================
        # DIRECTLY CALL MAPANYTHING INTERNALS (same as geometric DPT)
        # ============================================================
        # This replicates the exact feature extraction from MapAnything.forward()
        # to ensure we get the same features the geometric DPT head uses.
        
        with torch.no_grad():
            # Step 1: Encode images through DINOv2 using MapAnything's method
            # Prepare views in the format MapAnything expects
            views = [{
                'img': x,  # [B, 3, H, W]
                'data_norm_type': ['dinov2'] * batch_size,
            }]
            
            # Debug: print input image shape
            if not hasattr(self, '_printed_input_shape'):
                self._printed_input_shape = True
                print(f"\n[DEBUG] Input image shape: {x.shape}")
                print(f"[DEBUG] Expected patch grid: {padded_h // self.patch_size} x {padded_w // self.patch_size}")
            
            # Use MapAnything's internal method to encode views properly
            all_encoder_features_across_views = self.mapanything._encode_n_views(views)
            # Result is a tuple/list with one tensor per view, each [B, C, H, W]
            encoder_features = all_encoder_features_across_views[0]  # Single view: [B, C, H, W]
            
            # Debug: print encoder output shape
            if not hasattr(self, '_printed_encoder_shape'):
                self._printed_encoder_shape = True
                print(f"[DEBUG] Encoder output shape: {encoder_features.shape}")
                print(f"[DEBUG] Encoder output type: {type(encoder_features)}")
                if hasattr(self.mapanything.encoder, 'patch_size'):
                    print(f"[DEBUG] Encoder patch_size: {self.mapanything.encoder.patch_size}")
                # Check if encoder has other attributes
                print(f"[DEBUG] Encoder type: {type(self.mapanything.encoder)}")
            
            # Step 2: Normalize encoder features for fusion
            # MapAnything permutes to [B, H, W, C] for LayerNorm, then back to [B, C, H, W]
            encoder_features_permuted = encoder_features.permute(0, 2, 3, 1).contiguous()
            fused_features = self.mapanything.fusion_norm_layer(
                encoder_features_permuted.float()
            ).to(encoder_features.dtype)
            fused_features = fused_features.permute(0, 3, 1, 2).contiguous()  # Back to [B, C, H, W]
            
            # Step 3: Pass through info_sharing transformer
            # Prepare scale token (same as MapAnything.forward)
            input_scale_token = (
                self.mapanything.scale_token.unsqueeze(0)
                .unsqueeze(-1)
                .repeat(batch_size, 1, 1)
            )  # (B, C, 1)
            
            # Create transformer input - features is a list (one per view)
            # Each feature tensor is [B, C, H, W]
            info_sharing_input = MultiViewTransformerInput(
                features=[fused_features],  # List with single view, each [B, C, H, W]
                additional_input_tokens=input_scale_token,
            )
            
            # Call info_sharing - returns (final, [intermediates]) for IFR variant
            info_sharing_output = self.mapanything.info_sharing(info_sharing_input)
            
            if isinstance(info_sharing_output, tuple) and len(info_sharing_output) == 2:
                final_features_output, intermediate_features_list = info_sharing_output
            else:
                final_features_output = info_sharing_output
                intermediate_features_list = []
            
            # Step 4: Extract the 4 layers of features for DPT
            # Following MapAnything's pattern exactly:
            # [encoder_features, intermediate[0], intermediate[1], final]
            
            layer_features = []
            
            if self.use_encoder_features_for_dpt:
                # Layer 0: Encoder features [B, num_tokens, 1024]
                layer_features.append(encoder_features)
                
                # Layers 1-2: Intermediate features from info_sharing
                for i, intermediate_output in enumerate(intermediate_features_list):
                    if hasattr(intermediate_output, 'features'):
                        # .features is a list (one per view), take first view
                        feat = intermediate_output.features[0]
                    else:
                        feat = intermediate_output[0] if isinstance(intermediate_output, (list, tuple)) else intermediate_output
                    layer_features.append(feat)
                
                # Layer 3: Final features from info_sharing
                if hasattr(final_features_output, 'features'):
                    final_feat = final_features_output.features[0]
                else:
                    final_feat = final_features_output[0] if isinstance(final_features_output, (list, tuple)) else final_features_output
                layer_features.append(final_feat)
            else:
                # All 4 layers from info_sharing (no encoder features)
                for intermediate_output in intermediate_features_list:
                    if hasattr(intermediate_output, 'features'):
                        feat = intermediate_output.features[0]
                    else:
                        feat = intermediate_output[0] if isinstance(intermediate_output, (list, tuple)) else intermediate_output
                    layer_features.append(feat)
                
                if hasattr(final_features_output, 'features'):
                    final_feat = final_features_output.features[0]
                else:
                    final_feat = final_features_output[0] if isinstance(final_features_output, (list, tuple)) else final_features_output
                layer_features.append(final_feat)
        
        # Ensure we have 4 layers of features
        if len(layer_features) < 4:
            print(f"WARNING: Only captured {len(layer_features)} layer features, expected 4")
            for i, f in enumerate(layer_features):
                if f is not None:
                    print(f"  Layer {i}: shape={f.shape if hasattr(f, 'shape') else type(f)}")
            # Pad with the last available feature if needed
            while len(layer_features) < 4:
                if layer_features:
                    layer_features.append(layer_features[-1])
                else:
                    patch_h, patch_w = padded_h // self.patch_size, padded_w // self.patch_size
                    layer_features.append(torch.zeros(batch_size, patch_h * patch_w, self.info_sharing_dim, device=x.device))
        
        # Take only first 4 if we have more
        layer_features = layer_features[:4]
        
        # Debug: print feature shapes on first iteration
        if not hasattr(self, '_printed_feature_shapes'):
            self._printed_feature_shapes = True
            print("\n[DEBUG] Feature shapes for Panoptic DPT:")
            for i, f in enumerate(layer_features):
                print(f"  Layer {i}: {f.shape} (expected dim: {self.panoptic_dpt.input_dims[i]})")
        
        # Run through Panoptic DPT head (trainable)
        outputs = self.panoptic_dpt(layer_features, padded_h, padded_w)
        
        # Crop outputs if we padded the input
        if pad_h > 0 or pad_w > 0:
            for key in outputs:
                stride = self._out_feature_strides[key]
                target_h = orig_h // stride
                target_w = orig_w // stride
                outputs[key] = outputs[key][:, :, :target_h, :target_w]
        
        return outputs
    
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
        """Ensure MapAnything stays frozen while training Panoptic DPT."""
        super().train(mode)
        self.mapanything.eval()
        for param in self.mapanything.parameters():
            param.requires_grad = False
        self.panoptic_dpt.train(mode)
        return self


# ============================================================
# REGISTER BACKBONE TO DETECTRON2
# ============================================================

from detectron2.modeling.backbone import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
def build_mapanything_panoptic_dpt_backbone(cfg, input_shape):
    return MapAnythingWithPanopticDPT(cfg, input_shape)


# ============================================================
# CUSTOM TRAINER WITH DIFFERENTIAL LEARNING RATES
# ============================================================

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
        - Panoptic DPT (copied from geometric): cfg.SOLVER.DPT_LR
        - Output projections (new): cfg.SOLVER.BASE_LR
        - Mask2Former head: cfg.SOLVER.BASE_LR
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
        params = []
        
        if dpt_params:
            params.append({
                'params': dpt_params,
                'lr': cfg.SOLVER.DPT_LR,
                'name': 'panoptic_dpt'
            })
        
        if projection_params:
            params.append({
                'params': projection_params,
                'lr': cfg.SOLVER.BASE_LR,
                'name': 'dpt_projections'
            })
        
        if other_params:
            params.append({
                'params': other_params,
                'lr': cfg.SOLVER.BASE_LR,
                'name': 'mask2former'
            })
        
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
        from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
            MaskFormerPanopticDatasetMapper,
        )
        
        # Data augmentations
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
        
        # NOTE: size_divisibility=0 disables the dataset mapper's padding.
        # The backbone handles its own padding to be divisible by patch_size.
        # The original mapper code has a bug that crops images when size_divisibility < image_size.
        mapper = MaskFormerPanopticDatasetMapper(
            is_train=True,
            augmentations=augmentations,
            image_format=cfg.INPUT.FORMAT,
            ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=0,  # Disable mapper padding - backbone handles this
        )
        
        # Optimize dataloader for multi-GPU training
        # With 4 GPUs, 8 workers per GPU provides good throughput without CPU bottleneck
        cfg.DATALOADER.NUM_WORKERS = 8
        
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def resume_or_load(self, resume=True):
        super().resume_or_load(resume=resume)
        
        model = self.model
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
            
        # Re-freeze MapAnything after loading checkpoint
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "mapanything"):
                print("Re-freezing MapAnything backbone parameters...")
                model.backbone.mapanything.eval()
                for param in model.backbone.mapanything.parameters():
                    param.requires_grad = False
                print("MapAnything re-frozen, Panoptic DPT trainable.")


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
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_panoptic_dpt_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    cfg.MODEL.MAPANYTHING = CN()
    cfg.MODEL.MAPANYTHING.CHECKPOINT_PATH = mapanything_checkpoint_path
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # ===== MASK2FORMER HEAD CONFIGURATION =====
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 133  # COCO panoptic
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
    cfg.SOLVER.DPT_LR = dpt_lr  # Differential LR for copied DPT weights
    cfg.SOLVER.MAX_ITER = max_iter
    
    # LR schedule: Single decay at ~80% of training (proportional to max_iter)
    # Original: decay at 70k/90k = 77.8% → now at 40k/50k = 80%
    cfg.SOLVER.STEPS = (40000,)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Not used (we handle LRs manually)
    
    # Warmup for stable training start
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
    cfg.DATALOADER.NUM_WORKERS = 8  # Set in build_train_loader, optimal for 4 GPUs
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.VERSION = 2
    
    print("\n" + "="*80)
    print("THREE-COMPONENT PIPELINE CONFIGURATION:")
    print("="*80)
    print("1. FROZEN MapAnything Backbone:")
    print(f"   - DINOv2 + Multi-View Transformer")
    print(f"   - Output: 768-dim features from 4 transformer layers")
    print("")
    print("2. TRAINABLE Panoptic DPT Head:")
    print(f"   - Initialized from geometric DPT weights")
    print(f"   - Learning rate: {dpt_lr} (DPT) / {learning_rate} (projections)")
    print(f"   - Output channels: res2(96), res3(192), res4(384), res5(768)")
    print("")
    print("3. TRAINABLE Mask2Former Head:")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Queries: 100, Decoder layers: 9")
    print(f"   - MSDeformAttn Pixel Decoder")
    print("")
    print("TRAINING SCHEDULE (Time-Optimized):")
    print(f"   - Max iterations: {max_iter} (reduced from 90k)")
    print(f"   - LR decay at: {cfg.SOLVER.STEPS[0]} iterations")
    print(f"   - Effective batch: {batch_size * num_gpus} ({batch_size}/GPU × {num_gpus} GPUs)")
    print(f"   - Estimated time: ~8.5 hours on 4× A100 80GB")
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
    dpt_lr: float = 1e-5,
    max_iter: int = 90000,
    resume: bool = False,
):
    print("="*80)
    print("MASK2FORMER + MAPANYTHING + PANOPTIC DPT - THREE COMPONENT PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  MapAnything checkpoint: {mapanything_checkpoint}")
    print(f"  COCO root: {coco_root}")
    print(f"  Output directory: {output_dir}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (Mask2Former + new projections)")
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
    train_panoptic_segmentation_head(
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
    
    # ===== RELATIVE PATH CONFIGURATION =====
    BASE_DIR = os.getcwd()
    
    # 1. MapAnything Checkpoint
    MAPANYTHING_CHECKPOINT = os.path.join(BASE_DIR, "pretrained_models", "map_anything", "test")
    
    # 2. COCO Dataset
    COCO_ROOT = os.path.join(BASE_DIR, "datasets", "coco")
    
    # 3. Output Directory
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_cluster")
    
    # ===== CLUSTER HYPERPARAMETERS =====
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Detected GPU: {gpu_name}")
    
    if "A100" in gpu_name:
        BATCH_SIZE = 4
        print(f"Configuring for A100 (80GB): Batch size {BATCH_SIZE}")
    else:
        BATCH_SIZE = 2  # V100 16GB can't handle batch_size=4 with full resolution
        print(f"Configuring for V100 (16GB): Batch size {BATCH_SIZE}")

    LEARNING_RATE = 1e-4  # For Mask2Former and new projections
    DPT_LR = 1e-5  # Lower LR for copied DPT weights
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

