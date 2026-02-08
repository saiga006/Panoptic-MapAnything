# Three-Component Pipeline Implementation Summary

## Architecture Overview

```
Input Image [B, 3, H, W]
    ↓
┌─────────────────────────────────────────┐
│  FROZEN: MapAnything Backbone           │
│  - DINOv2 Encoder (patch_size=14)      │
│  - Multi-View Transformer (N=1)        │
│  Output: aggregated_tokens_list        │
│          [B, N_tokens, 768] x 24 layers│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Panoptic DPT Head           │
│  - Initialized from Geometric DPT       │
│  - Projects layers [4, 11, 17, 23]     │
│  - Reassemble: [256→512→1024→1024]    │
│  - Fusion: 4 refinenet blocks          │
│  - Output Projections (NEW):           │
│    res2: 256→96   (stride 4)          │
│    res3: 256→192  (stride 8)          │
│    res4: 256→384  (stride 16)         │
│    res5: 256→768  (stride 32)         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Mask2Former Head            │
│  - MSDeformAttn Pixel Decoder           │
│  - Transformer Decoder (9 layers)       │
│  - Classification + Mask heads          │
│  Output: Panoptic Predictions           │
└─────────────────────────────────────────┘
```

## Key Implementation Details

### 1. Panoptic DPT Head (`PanopticDPTHead`)

**Purpose**: Convert 768-dim transformer tokens into multi-scale feature pyramid

**Architecture Components**:
- **Norm layer**: LayerNorm(768) - normalizes tokens before processing
- **Projects**: 4x Conv2d(768 → [256, 512, 1024, 1024]) - project tokens to intermediate channels
- **Resize layers**: 
  - Layer 0: ConvTranspose2d (4x upsample)
  - Layer 1: ConvTranspose2d (2x upsample)
  - Layer 2: Identity (same size)
  - Layer 3: Conv2d stride=2 (2x downsample)
- **Scratch layers**: 4x Conv2d(intermediate → 256) - unify to 256 channels
- **Refinenet blocks**: 4 fusion blocks (ResidualConvUnit + FeatureFusionBlock)
- **Output projections** (NEW): Conv2d(256 → [96, 192, 384, 768])

**Weight Initialization**:
- All components EXCEPT output_projections are copied from MapAnything's geometric DPT
- Output projections are randomly initialized (they're task-specific)

### 2. Frozen MapAnything Backbone (`FrozenMapAnythingBackbone`)

**Changes from original**:
- No longer hooks into DPT refinenet layers
- Outputs raw transformer tokens instead of processed features
- Returns: `{'tokens': aggregated_tokens_list, 'H': orig_h, 'W': orig_w}`
- All parameters frozen (requires_grad=False)

### 3. Combined Wrapper (`MapAnythingWithPanopticDPT`)

**Purpose**: Present unified interface to Detectron2

**Methods**:
- `__init__`: Creates both frozen backbone and trainable DPT, initializes weights
- `forward`: Chains backbone → DPT → returns feature pyramid
- `_initialize_panoptic_dpt_from_geometric`: Copies weights from geometric DPT
- `output_shape`: Returns ShapeSpec for res2/3/4/5

### 4. Differential Learning Rates

**Parameter Groups**:
1. **Panoptic DPT (copied weights)**: LR = 1e-5
   - norm, projects, resize_layers, scratch, refinenet blocks
2. **DPT Output Projections (new)**: LR = 1e-4
   - output_projections module
3. **Mask2Former Head**: LR = 1e-4
   - All sem_seg_head parameters

**Implementation**: Custom `build_optimizer` in `Mask2FormerPanopticTrainer`

## Configuration Highlights

```python
# Backbone
cfg.MODEL.BACKBONE.NAME = "build_mapanything_with_panoptic_dpt_backbone"
cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

# Output channels match Mask2Former requirements
# res2: 96 (stride 4)
# res3: 192 (stride 8)  
# res4: 384 (stride 16)
# res5: 768 (stride 32)

# Learning rates
cfg.SOLVER.BASE_LR = 1e-4  # Mask2Former + new projections
cfg.SOLVER.DPT_LR = 1e-5   # Copied DPT components

# Training schedule
cfg.SOLVER.MAX_ITER = 90000
cfg.SOLVER.STEPS = (70000,)  # Single LR decay
cfg.SOLVER.WARMUP_ITERS = 1000
```

## Training Strategy

### Phase A: Joint Training (Current Implementation)
- MapAnything backbone: **FROZEN**
- Panoptic DPT: **TRAINABLE** (LR = 1e-5 for copied, 1e-4 for new)
- Mask2Former: **TRAINABLE** (LR = 1e-4)

### Future Phase B (Optional):
Could fine-tune both DPT and Mask2Former together at same LR if needed

## Weight Copying Details

The `_initialize_panoptic_dpt_from_geometric` method copies:

1. **norm** layer weights
2. **projects** (4 Conv2d layers)
3. **resize_layers** (excluding Identity)
4. **scratch.layer{1,2,3,4}_rn** (4 Conv2d layers)
5. **scratch.refinenet{1,2,3,4}** blocks:
   - out_conv
   - resConfUnit1.conv1, resConfUnit1.conv2
   - resConfUnit2.conv1, resConfUnit2.conv2

**Not copied** (randomly initialized):
- output_projections (task-specific for panoptic segmentation)

## Memory Efficiency

- MapAnything backbone runs with `torch.no_grad()` → no gradient storage
- Only Panoptic DPT + Mask2Former consume GPU memory for gradients
- Estimated trainable parameters:
  - Panoptic DPT: ~10-15M parameters
  - Mask2Former: ~40-50M parameters
  - Total trainable: ~50-65M (vs ~200M if unfrozen backbone)

## Usage

```bash
# Run training
cd /home/saiga/mnt/Mask2Former
python m2f_train_panoptic_dpt.py --num-gpus 1

# Multi-GPU training
python m2f_train_panoptic_dpt.py --num-gpus 4
```

## Expected Improvements

1. **Better feature initialization**: DPT starts with geometric priors instead of random weights
2. **Stable training**: Lower LR for pretrained DPT prevents catastrophic forgetting
3. **Semantic alignment**: DPT learns to adapt geometric features for semantic tasks
4. **Multi-scale quality**: Proper pyramid construction at each level

## Key Differences from Previous Implementation

| Aspect | Old (m2f_train_cluster_working.py) | New (m2f_train_panoptic_dpt.py) |
|--------|-------------------------------------|----------------------------------|
| DPT Head | Hooks into frozen DPT | Trainable duplicate of DPT |
| Weight Init | Random projections | Copied from geometric DPT |
| Output Channels | Uniform 256 → adapted | Multi-scale 96/192/384/768 |
| Learning Rate | Single LR for all | Differential (1e-5 vs 1e-4) |
| Architecture | 2 components | 3 components |
| Feature Flow | Backbone → Mask2Former | Backbone → DPT → Mask2Former |

## Verification Checklist

- [x] PanopticDPTHead implements full DPT architecture
- [x] Weight copying from geometric DPT works
- [x] Backbone outputs raw tokens (not DPT features)
- [x] Multi-scale pyramid has correct channel dimensions
- [x] Differential learning rates configured
- [x] Mask2Former receives expected input shapes
- [x] All frozen parameters excluded from optimizer
- [x] Output projections use higher LR than copied weights
