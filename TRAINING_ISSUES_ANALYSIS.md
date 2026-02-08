# Training Issues Analysis - Mask2Former + MapAnything

## Current Performance (90k iterations)
- **Loss:** Stagnated at ~30
- **PQ (Panoptic Quality):** 0.061 (extremely low)
- **Things PQ:** 0.000 (complete failure to detect objects)
- **Stuff PQ:** 0.153 (minimal segmentation)

---

## ROOT CAUSES IDENTIFIED

### 🔴 CRITICAL ISSUE #1: No Data Augmentation
**Location:** Line 297-302 in `build_train_loader()`

```python
mapper = MaskFormerPanopticDatasetMapper(
    is_train=True,
    augmentations=[],  # ← EMPTY AUGMENTATIONS!
    ...
)
```

**Impact:**
- Model sees identical crops every epoch
- Severe overfitting to specific image orientations/scales
- Cannot generalize to validation set
- Standard Mask2Former uses extensive augmentations (flip, resize, crop)

**Fix Required:**
```python
from detectron2.data import transforms as T

augmentations = [
    T.ResizeShortestEdge(
        short_edge_length=(640, 672, 704, 736, 768, 800),
        max_size=1333,
        sample_style="choice"
    ),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
]
```

---

### 🔴 CRITICAL ISSUE #2: Frozen Backbone Problem
**Location:** Lines 93-96, 319-325

**Current Setup:**
- MapAnything backbone: **100% FROZEN**
- Only trainable: Small 1x1 conv projection layers (~few thousand params)
- Total trainable params: < 1% of model

**Impact:**
- Backbone features are optimized for 3D reconstruction, NOT segmentation
- Projection layers alone cannot bridge this semantic gap
- Model cannot adapt features to panoptic segmentation task

**Evidence from Results:**
- SQ (Segmentation Quality) = 0.523 suggests masks aren't terrible
- RQ (Recognition Quality) = 0.088 shows classification is failing
- This indicates feature representations are wrong for this task

**Solutions (pick ONE):**

**Option A: Unfreeze Last Backbone Layers**
```python
# In FrozenMapAnythingBackbone.__init__()
# Freeze only early layers, unfreeze refinenet layers
for name, param in self.mapanything.named_parameters():
    if 'refinenet' in name:  # Unfreeze refinenet layers
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**Option B: Use Deeper Projection Adapters**
```python
def _create_projection_layers(self):
    """Create trainable projection layers with deeper capacity"""
    self.projections = nn.ModuleDict()
    dpt_channels = 256
    
    for feat_name, target_channels in self._out_feature_channels.items():
        # Use 3-layer MLP instead of 1x1 conv
        self.projections[feat_name] = nn.Sequential(
            nn.Conv2d(dpt_channels, dpt_channels, 3, padding=1),
            nn.GroupNorm(32, dpt_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dpt_channels, target_channels, 3, padding=1),
            nn.GroupNorm(32, target_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels, target_channels, 1)
        )
```

---

### 🟡 ISSUE #3: Learning Rate Schedule
**Location:** Lines 413-420

```python
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.STEPS = (60000, 80000)  # LR drops at 60k, 80k
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1  # ← Useless when backbone frozen
```

**Problems:**
- Base LR might be too low for training from scratch
- LR schedule (60k, 80k) with GAMMA=0.1 means:
  - 0-60k: LR = 1e-4
  - 60k-80k: LR = 1e-5 
  - 80k-90k: LR = 1e-6 (too small!)
- By 90k iterations, LR is 1e-6, explaining why loss stopped improving

**Fix:**
```python
cfg.SOLVER.BASE_LR = 1e-4  # OK if backbone frozen, increase to 2e-4 if unfrozen
cfg.SOLVER.STEPS = (70000,)  # Single drop closer to end
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_FACTOR = 0.001
```

---

### 🟡 ISSUE #4: Batch Size Too Small
**Location:** Lines 608-613

```python
if "A100" in gpu_name:
    BATCH_SIZE = 8  # Per GPU
else:
    BATCH_SIZE = 4  # Per GPU
```

**For Panoptic Segmentation:**
- Mask2Former typically uses batch size 16-32 globally
- Small batch = unstable gradients for transformer decoder
- Small batch = poor batch normalization statistics

**Fix (if you have multiple GPUs):**
```python
# If using 4 GPUs with A100
BATCH_SIZE = 8  # 8 x 4 = 32 total (good)

# If using 1 GPU
BATCH_SIZE = 16  # Try to push higher if memory allows
```

---

### 🟡 ISSUE #5: Loss Weight Imbalance
**Location:** Lines 378-382

```python
cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 2.0
cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
```

**Analysis:**
- CLASS_WEIGHT = 2.0 is low compared to MASK/DICE weights
- This might explain why Things PQ = 0.0 (classification failing)
- Model prioritizes mask quality over correct classification

**Suggested Fix:**
```python
cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 5.0  # Increase from 2.0
cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0
cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 5.0
```

---

### 🟡 ISSUE #6: Input Resolution Mismatch
**Location:** Line 402

```python
cfg.INPUT.IMAGE_SIZE = 1024
```

**Potential Problem:**
- MapAnything was pretrained on specific resolutions
- 1024x1024 might not match MapAnything's expected input
- Fixed size = no scale variation during training

---

## PRIORITY ACTION PLAN

### **Phase 1: Quick Wins (Test Immediately)**
1. ✅ Add data augmentations (flip, multi-scale resize)
2. ✅ Increase CLASS_WEIGHT from 2.0 to 5.0
3. ✅ Adjust LR schedule: STEPS = (70000,)
4. ✅ Add warmup: WARMUP_ITERS = 1000

### **Phase 2: Architecture Changes (If Phase 1 Fails)**
5. ✅ Replace 1x1 conv projections with deeper 3-layer MLPs
6. ✅ Unfreeze MapAnything refinenet layers
7. ✅ Increase batch size if GPU memory allows

### **Phase 3: Advanced (If Still Not Working)**
8. ⚠️ Train a baseline Mask2Former with ResNet50 backbone on same data
9. ⚠️ Compare loss curves to diagnose if MapAnything features are compatible
10. ⚠️ Consider fine-tuning MapAnything on COCO before freezing

---

## EXPECTED IMPROVEMENTS

### After Phase 1 (Augmentations + Hyperparams):
- **Loss:** Should drop to 10-15 range
- **Things PQ:** Should be > 0.20
- **Stuff PQ:** Should be > 0.30
- **Overall PQ:** Should reach 0.25-0.30

### After Phase 2 (Architecture):
- **Loss:** Should drop to 5-8 range
- **Overall PQ:** Should reach 0.35-0.40

### Baseline Mask2Former (for reference):
- **Overall PQ:** 0.45-0.50 on COCO val

---

## DIAGNOSTIC COMMANDS

```bash
# Check model parameters
grep "Total frozen parameters" output_cluster/log.txt

# Check training loss curve
grep "total_loss" output_cluster/log.txt | tail -100

# Check learning rate over time
grep "lr:" output_cluster/log.txt | tail -50

# Monitor GPU memory
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

---

## QUESTIONS TO INVESTIGATE

1. **What resolution was MapAnything pretrained on?**
   - Check MapAnything config/paper
   - Match INPUT.IMAGE_SIZE to this

2. **Are MapAnything features aligned to COCO classes?**
   - MapAnything was trained for 3D reconstruction
   - May need semantic alignment layer

3. **Is the feature channel projection correct?**
   - Verify res2/3/4/5 channel counts match Mask2Former expectations
   - Standard FPN expects [256, 512, 1024, 2048]

---

## NEXT STEPS

1. Run `m2f_train_cluster_working_FIXED.py` (to be created with Phase 1 fixes)
2. Monitor first 5000 iterations - loss should drop below 20
3. Check 10k iteration eval - Things PQ should be > 0
4. If still failing, proceed to Phase 2 changes

