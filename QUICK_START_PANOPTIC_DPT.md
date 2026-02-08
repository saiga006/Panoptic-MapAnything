# Quick Start Guide - Three-Component Panoptic DPT Pipeline

## Overview

This pipeline combines:
1. **Frozen MapAnything Backbone** - Provides geometric priors from 3D reconstruction
2. **Trainable Panoptic DPT** - Adapts geometric features to semantic features (initialized from geometric DPT)
3. **Trainable Mask2Former** - Query-based panoptic segmentation

## Files Created

- `m2f_train_panoptic_dpt.py` - Main training script
- `verify_panoptic_dpt.py` - Architecture verification script
- `PANOPTIC_DPT_PIPELINE.md` - Detailed architecture documentation

## Step-by-Step Usage

### 1. Verify Architecture (Recommended First)

```bash
cd /home/saiga/mnt/Mask2Former
python verify_panoptic_dpt.py
```

This will check:
- Model builds correctly
- MapAnything is frozen
- Panoptic DPT is trainable with correct weight initialization
- Mask2Former is trainable
- Output shapes are correct (res2: 96, res3: 192, res4: 384, res5: 768 channels)
- Forward pass works

### 2. Run Training

#### Single GPU
```bash
python m2f_train_panoptic_dpt.py --num-gpus 1
```

#### Multi-GPU (4 GPUs)
```bash
python m2f_train_panoptic_dpt.py --num-gpus 4
```

#### With Custom Paths
```bash
python m2f_train_panoptic_dpt.py \
    --num-gpus 2 \
    --config-file configs/custom_config.yaml \
    OUTPUT_DIR ./my_output
```

### 3. Monitor Training

```bash
# Watch tensorboard
tensorboard --logdir output_panoptic_dpt

# Check latest checkpoint
ls -lth output_panoptic_dpt/*.pth | head -5
```

## Key Configuration Parameters

### Learning Rates
- **Panoptic DPT (copied weights)**: `1e-5` (lower to preserve geometric knowledge)
- **DPT Output Projections (new)**: `1e-4` (higher for faster adaptation)
- **Mask2Former**: `1e-4` (standard)

### Batch Sizes
- **V100 (16GB)**: Batch size 4
- **A100 (80GB)**: Batch size 8
- Automatically detected based on GPU name

### Training Schedule
- **Max iterations**: 90,000
- **LR decay**: Single step at 70,000 iterations (0.1x)
- **Warmup**: 1,000 iterations (from 1e-7)
- **Evaluation**: Every 5,000 iterations
- **Checkpoints**: Every 5,000 iterations

## Architecture Summary

```
Input [B, 3, H, W]
    ↓
MapAnything (FROZEN)
    ↓ 768-dim tokens @ H/14
Panoptic DPT (TRAINABLE, initialized from geometric)
    ↓ Multi-scale pyramid
    res2: [B, 96, H/4, W/4]
    res3: [B, 192, H/8, W/8]
    res4: [B, 384, H/16, W/16]
    res5: [B, 768, H/32, W/32]
    ↓
Mask2Former (TRAINABLE)
    ↓
Panoptic Predictions
```

## Weight Initialization Details

### Copied from Geometric DPT:
- ✅ Token normalization layer
- ✅ Token projection layers (4 layers)
- ✅ Resize layers (upsample/downsample)
- ✅ Scratch adaptation layers (4 Conv2d)
- ✅ Refinenet fusion blocks (4 blocks with ResidualConvUnits)

### Randomly Initialized:
- 🎲 Output projection layers (res2/3/4/5) - Task-specific for panoptic

## Troubleshooting

### Issue: MapAnything checkpoint not found
```bash
# Verify path
ls -la pretrained_models/map_anything/test/

# Expected files:
# - config.json
# - pytorch_model.bin or model.safetensors
```

### Issue: COCO dataset not found
```bash
# Verify COCO structure
ls -la datasets/coco/
# Should contain:
# - train2017/
# - val2017/
# - annotations/panoptic_train2017.json
# - annotations/panoptic_val2017.json
# - panoptic_train2017/
# - panoptic_val2017/
```

### Issue: CUDA out of memory
```python
# Reduce batch size in script:
BATCH_SIZE = 2  # Instead of 4

# Or use gradient accumulation (manual modification needed)
```

### Issue: Weight copying warnings
```
# This is normal if geometric DPT has slightly different architecture
# Verify with verification script first
python verify_panoptic_dpt.py
```

## Expected Training Behavior

### First 1000 iterations (Warmup)
- Loss should decrease from ~10 to ~5
- Learning rate ramps from 1e-7 to configured values

### Iterations 1000-70000
- Steady loss decrease
- Classification loss should balance with mask loss (CLASS_WEIGHT=5.0)
- PQ should gradually improve on validation

### After iteration 70000 (LR decay)
- Learning rate drops 10x
- Fine-tuning phase
- Slower but steadier improvements

## Performance Monitoring

### Key Metrics to Watch
- **PQ (Panoptic Quality)**: Primary metric for panoptic segmentation
- **SQ (Segmentation Quality)**: Mask quality
- **RQ (Recognition Quality)**: Classification quality
- **Total Loss**: Should steadily decrease
- **Class Loss**: Should not dominate (balanced by CLASS_WEIGHT=5.0)

### Good Training Signs
- ✅ Loss decreases smoothly
- ✅ PQ increases over time
- ✅ No NaN/Inf losses
- ✅ Validation metrics improve

### Bad Training Signs
- ❌ Loss explodes or oscillates wildly
- ❌ NaN/Inf losses (will auto-stop with NaNLossCheckHook)
- ❌ Class loss dominates mask loss
- ❌ No improvement after 20k iterations

## Comparison with Previous Approach

| Aspect | Old (working script) | New (Panoptic DPT) |
|--------|---------------------|---------------------|
| DPT | Frozen (hooks) | Trainable (duplicate) |
| Weight Init | Random | Copied from geometric |
| Channels | Uniform 256 | Multi-scale 96/192/384/768 |
| Learning Rate | Single | Differential (1e-5/1e-4) |
| Components | 2 | 3 |
| Feature Transfer | Direct | Through trainable DPT |

## Next Steps After Training

### Evaluate Model
```bash
# Run evaluation on validation set
python m2f_train_panoptic_dpt.py \
    --eval-only \
    MODEL.WEIGHTS output_panoptic_dpt/model_final.pth
```

### Inference on Custom Images
```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
# ... load config ...
cfg.MODEL.WEIGHTS = "output_panoptic_dpt/model_final.pth"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

### Visualize Predictions
```bash
python demo/demo.py \
    --config-file configs/coco/panoptic-segmentation/... \
    --input path/to/images/*.jpg \
    --output path/to/output \
    --opts MODEL.WEIGHTS output_panoptic_dpt/model_final.pth
```

## Contact & Support

For issues specific to:
- **Architecture**: Check `PANOPTIC_DPT_PIPELINE.md`
- **Training**: Monitor logs in `output_panoptic_dpt/log.txt`
- **Verification**: Run `python verify_panoptic_dpt.py`

## References

- DPT Paper: https://arxiv.org/abs/2103.13413
- Mask2Former: https://arxiv.org/abs/2112.01527
- MapAnything: [Project documentation]
