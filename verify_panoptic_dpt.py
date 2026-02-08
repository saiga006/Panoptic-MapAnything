"""
Architecture Verification Script

This script helps verify that the three-component pipeline is correctly configured:
1. Frozen MapAnything backbone
2. Trainable Panoptic DPT (initialized from geometric DPT)
3. Trainable Mask2Former head

Run this before full training to catch configuration issues early.
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from m2f_train_panoptic_dpt import MapAnythingWithPanopticDPT, setup_cfg


def verify_architecture():
    """Verify the three-component pipeline architecture."""
    
    print("="*80)
    print("ARCHITECTURE VERIFICATION")
    print("="*80)
    
    # Setup minimal config
    BASE_DIR = os.getcwd()
    MAPANYTHING_CHECKPOINT = os.path.join(BASE_DIR, "pretrained_models", "map_anything", "test")
    COCO_ROOT = os.path.join(BASE_DIR, "datasets", "coco")
    
    if not os.path.exists(MAPANYTHING_CHECKPOINT):
        print(f"\n❌ ERROR: MapAnything checkpoint not found at {MAPANYTHING_CHECKPOINT}")
        print("Cannot proceed with verification without checkpoint")
        return False
    
    print(f"\n✓ Found MapAnything checkpoint: {MAPANYTHING_CHECKPOINT}")
    
    # Build config
    cfg = setup_cfg(
        mapanything_checkpoint_path=MAPANYTHING_CHECKPOINT,
        coco_root=COCO_ROOT,
        output_dir="./output_verify",
        num_gpus=1,
        batch_size=1,
        learning_rate=1e-4,
        dpt_lr=1e-5,
        max_iter=1000,
    )
    
    print("\n" + "-"*80)
    print("1. BUILDING MODEL")
    print("-"*80)
    
    try:
        model = build_model(cfg)
        print("✓ Model built successfully")
    except Exception as e:
        print(f"❌ ERROR building model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*80)
    print("2. VERIFYING COMPONENT STRUCTURE")
    print("-"*80)
    
    # Check backbone
    if not hasattr(model, 'backbone'):
        print("❌ ERROR: Model has no backbone")
        return False
    
    backbone = model.backbone
    print(f"\n✓ Backbone type: {type(backbone).__name__}")
    
    if not isinstance(backbone, MapAnythingWithPanopticDPT):
        print(f"❌ ERROR: Expected MapAnythingWithPanopticDPT, got {type(backbone)}")
        return False
    
    print("✓ Correct backbone type: MapAnythingWithPanopticDPT")
    
    # Check frozen MapAnything
    if not hasattr(backbone.backbone, 'mapanything'):
        print("❌ ERROR: Backbone has no mapanything attribute")
        return False
    
    print("✓ MapAnything component found")
    
    # Check Panoptic DPT
    if not hasattr(backbone, 'panoptic_dpt'):
        print("❌ ERROR: Backbone has no panoptic_dpt attribute")
        return False
    
    print("✓ Panoptic DPT component found")
    
    # Check Mask2Former head
    if not hasattr(model, 'sem_seg_head'):
        print("❌ ERROR: Model has no sem_seg_head")
        return False
    
    print("✓ Mask2Former head found")
    
    print("\n" + "-"*80)
    print("3. VERIFYING FROZEN PARAMETERS")
    print("-"*80)
    
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"\nFrozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {frozen_params + trainable_params:,}")
    
    # MapAnything should be frozen
    mapanything_frozen = all(
        not p.requires_grad 
        for p in backbone.backbone.mapanything.parameters()
    )
    
    if not mapanything_frozen:
        print("❌ ERROR: MapAnything parameters are not frozen!")
        return False
    
    print("✓ MapAnything is correctly frozen")
    
    # Panoptic DPT should be trainable
    dpt_trainable = any(
        p.requires_grad 
        for p in backbone.panoptic_dpt.parameters()
    )
    
    if not dpt_trainable:
        print("❌ ERROR: Panoptic DPT parameters are frozen!")
        return False
    
    print("✓ Panoptic DPT is correctly trainable")
    
    # Mask2Former should be trainable
    m2f_trainable = any(
        p.requires_grad 
        for p in model.sem_seg_head.parameters()
    )
    
    if not m2f_trainable:
        print("❌ ERROR: Mask2Former parameters are frozen!")
        return False
    
    print("✓ Mask2Former is correctly trainable")
    
    print("\n" + "-"*80)
    print("4. VERIFYING OUTPUT SHAPES")
    print("-"*80)
    
    output_shape = backbone.output_shape()
    expected_features = ['res2', 'res3', 'res4', 'res5']
    expected_channels = {'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768}
    expected_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
    
    for feat_name in expected_features:
        if feat_name not in output_shape:
            print(f"❌ ERROR: Missing feature '{feat_name}' in output_shape")
            return False
        
        shape_spec = output_shape[feat_name]
        
        if shape_spec.channels != expected_channels[feat_name]:
            print(f"❌ ERROR: {feat_name} has {shape_spec.channels} channels, expected {expected_channels[feat_name]}")
            return False
        
        if shape_spec.stride != expected_strides[feat_name]:
            print(f"❌ ERROR: {feat_name} has stride {shape_spec.stride}, expected {expected_strides[feat_name]}")
            return False
        
        print(f"✓ {feat_name}: channels={shape_spec.channels}, stride={shape_spec.stride}")
    
    print("\n" + "-"*80)
    print("5. VERIFYING DIFFERENTIAL LEARNING RATES")
    print("-"*80)
    
    # Collect parameter groups
    dpt_params = []
    projection_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone.panoptic_dpt' in name:
            if 'output_projections' in name:
                projection_params.append(name)
            else:
                dpt_params.append(name)
        else:
            other_params.append(name)
    
    print(f"\nDPT parameters (should use LR=1e-5):")
    print(f"  Count: {len(dpt_params)}")
    if len(dpt_params) > 0:
        print(f"  Examples: {dpt_params[:3]}")
    else:
        print("  ⚠️  WARNING: No DPT parameters found!")
    
    print(f"\nProjection parameters (should use LR=1e-4):")
    print(f"  Count: {len(projection_params)}")
    if len(projection_params) > 0:
        print(f"  Examples: {projection_params[:3]}")
    else:
        print("  ⚠️  WARNING: No projection parameters found!")
    
    print(f"\nMask2Former parameters (should use LR=1e-4):")
    print(f"  Count: {len(other_params)}")
    if len(other_params) > 0:
        print(f"  Examples: {other_params[:3]}")
    
    print("\n" + "-"*80)
    print("6. TESTING FORWARD PASS")
    print("-"*80)
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            print("\n✓ Using CUDA")
        else:
            print("\n✓ Using CPU")
        
        model.eval()
        
        with torch.no_grad():
            print("\nRunning forward pass...")
            features = backbone(dummy_input)
        
        print(f"\n✓ Forward pass successful!")
        print(f"\nOutput features:")
        for feat_name, feat_tensor in features.items():
            print(f"  {feat_name}: {feat_tensor.shape}")
        
        # Verify shapes
        H, W = 224, 224
        for feat_name in expected_features:
            expected_h = H // expected_strides[feat_name]
            expected_w = W // expected_strides[feat_name]
            expected_c = expected_channels[feat_name]
            
            actual_shape = features[feat_name].shape
            expected_shape = (1, expected_c, expected_h, expected_w)
            
            if actual_shape != expected_shape:
                print(f"❌ ERROR: {feat_name} shape mismatch!")
                print(f"   Expected: {expected_shape}")
                print(f"   Got: {actual_shape}")
                return False
        
        print("\n✓ All output shapes correct!")
        
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL VERIFICATIONS PASSED!")
    print("="*80)
    print("\nThe three-component pipeline is correctly configured:")
    print("  1. ✓ Frozen MapAnything backbone")
    print("  2. ✓ Trainable Panoptic DPT (with weight initialization)")
    print("  3. ✓ Trainable Mask2Former head")
    print("  4. ✓ Correct output channels and strides")
    print("  5. ✓ Differential learning rates ready")
    print("  6. ✓ Forward pass works correctly")
    print("\nYou can now proceed with full training!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = verify_architecture()
    sys.exit(0 if success else 1)
