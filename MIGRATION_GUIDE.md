# Migration Guide: From Working Script to Panoptic DPT Pipeline

## Overview of Changes

This document explains the key architectural changes between the original working script (`m2f_train_cluster_working.py`) and the new three-component pipeline (`m2f_train_panoptic_dpt.py`).

## Architectural Transformation

### OLD: Two-Component Pipeline (Working Script)
```
Input → MapAnything (frozen, with DPT) → [hooked features] → Projections → Mask2Former
```

**Characteristics:**
- MapAnything completely frozen (including DPT)
- Hooks extract features from DPT refinenet layers
- Simple projection layers (3-layer MLPs) adapt channels
- All projections randomly initialized
- Single learning rate for all trainable components

### NEW: Three-Component Pipeline (Panoptic DPT)
```
Input → MapAnything (frozen, no DPT) → [tokens] → Panoptic DPT (trainable) → Mask2Former
```

**Characteristics:**
- MapAnything frozen (DINOv2 + transformer only)
- Panoptic DPT is separate, trainable module
- DPT duplicated from geometric version
- Weights initialized from geometric DPT
- Differential learning rates (1e-5 for DPT, 1e-4 for rest)

## Code-Level Changes

### 1. Backbone Class

#### OLD: `FrozenMapAnythingBackbone`
```python
class FrozenMapAnythingBackbone(Backbone):
    def _register_dpt_hooks(self):
        # Hooks into refinenet4, refinenet3, refinenet2, refinenet1
        mapping = {
            'refinenet4': 'res2',
            'refinenet3': 'res3',
            'refinenet2': 'res4',
            'refinenet1': 'res5'
        }
        # Extract features during forward pass
    
    def forward(self, x):
        # Run MapAnything, hooks capture features
        # Project features through trainable adapters
        return output_features
```

#### NEW: `FrozenMapAnythingBackbone` + `PanopticDPTHead`
```python
class FrozenMapAnythingBackbone(Backbone):
    # NO hooks - just extract tokens
    
    def forward(self, x):
        # Run MapAnything
        aggregated_tokens_list = output['aggregated_tokens_list']
        return {'tokens': aggregated_tokens_list, 'H': H, 'W': W}

class PanopticDPTHead(nn.Module):
    # Full DPT architecture duplicated
    def __init__(...):
        # Projects, resize_layers, scratch, refinenet blocks
        # Output projections
    
    def forward(self, tokens, H, W):
        # Process tokens through DPT
        return multi_scale_features
```

### 2. Channel Dimensions

#### OLD: Uniform 256 Channels
```python
self._out_feature_channels = {
    'res2': 256,
    'res3': 256,
    'res4': 256,
    'res5': 256,
}
```

#### NEW: Progressive Channels (Mask2Former Standard)
```python
output_channels = [96, 192, 384, 768]  # res2, res3, res4, res5
```

### 3. Weight Initialization

#### OLD: Random Projections
```python
self.projections = nn.ModuleDict()
for feat_name, target_channels in self._out_feature_channels.items():
    # 3-layer MLP with BatchNorm and ReLU
    self.projections[feat_name] = nn.Sequential(
        nn.Conv2d(256, 512, ...),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        # ... more layers
    )
    # All randomly initialized
```

#### NEW: Copied from Geometric DPT
```python
def _initialize_panoptic_dpt_from_geometric(self):
    geometric_dpt = self.backbone.mapanything.dpt_feature_head
    
    # Copy norm, projects, resize_layers, scratch, refinenet
    copy_weights(geometric_dpt.norm, panoptic_dpt.norm, "norm")
    for i in range(4):
        copy_weights(geometric_dpt.projects[i], panoptic_dpt.projects[i], ...)
    # ... etc
    
    # Only output_projections are randomly initialized
```

### 4. Learning Rate Configuration

#### OLD: Single Learning Rate
```python
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1  # Not really used (backbone frozen)
```

#### NEW: Differential Learning Rates
```python
cfg.SOLVER.BASE_LR = 1e-4      # Mask2Former + new projections
cfg.SOLVER.DPT_LR = 1e-5       # Copied DPT components

# In Trainer:
params = [
    {'params': dpt_params, 'lr': 1e-5},
    {'params': projection_params, 'lr': 1e-4},
    {'params': mask2former_params, 'lr': 1e-4}
]
```

### 5. Forward Pass Flow

#### OLD
```python
def forward(self, x):
    # MapAnything forward with hooks
    with torch.no_grad():
        _ = self.mapanything(views)  # Hooks capture features
    
    # Project captured features
    for feat_name in ['res2', 'res3', 'res4', 'res5']:
        feat = self.features[feat_name]
        output_features[feat_name] = self.projections[feat_name](feat)
    
    return output_features
```

#### NEW
```python
def forward(self, x):
    # 1. Get tokens from frozen MapAnything
    backbone_output = self.backbone(x)
    tokens = backbone_output['tokens']
    
    # 2. Process through trainable Panoptic DPT
    features = self.panoptic_dpt(tokens, H, W)
    
    return features  # Already at correct channels
```

## Feature Comparison

| Feature | Working Script | Panoptic DPT |
|---------|---------------|--------------|
| **MapAnything Role** | Frozen feature extractor | Frozen token provider |
| **DPT Architecture** | Frozen (hooked) | Trainable (duplicated) |
| **Weight Transfer** | None (random init) | Copied from geometric DPT |
| **Output Channels** | Uniform 256 | Progressive 96/192/384/768 |
| **Learning Strategy** | Single LR | Differential LR |
| **Trainable Params** | ~5-10M (projections) | ~50-65M (DPT + M2F) |
| **Memory Usage** | Lower | Moderate (more gradients) |
| **Training Stability** | Good | Better (pretrained DPT) |
| **Semantic Adaptation** | Direct projection | Learned through DPT |

## Performance Expectations

### Working Script
- ✅ Stable training from the start
- ✅ Lower memory footprint
- ⚠️ Random projection weights may need more iterations
- ⚠️ Uniform channels may not be optimal for Mask2Former

### Panoptic DPT Pipeline
- ✅ Better initialization (geometric priors)
- ✅ Proper multi-scale features
- ✅ Semantic features learned from geometric features
- ⚠️ Higher memory usage
- ⚠️ More complex architecture

## When to Use Which

### Use Working Script If:
- Limited GPU memory (<16GB)
- Want faster experimentation
- Need simpler architecture
- Don't have access to geometric DPT weights

### Use Panoptic DPT Pipeline If:
- Have sufficient GPU memory (≥16GB)
- Want better feature initialization
- Need proper multi-scale pyramid
- Want to leverage geometric priors from MapAnything
- Willing to train more parameters for better results

## Migration Checklist

If migrating from working script to Panoptic DPT:

- [ ] Update checkpoint paths in new script
- [ ] Verify MapAnything model has `dpt_feature_head` attribute
- [ ] Test with verification script first: `python verify_panoptic_dpt.py`
- [ ] Adjust batch size if needed (may need to reduce)
- [ ] Monitor first 1000 iterations for stability
- [ ] Compare PQ metrics after ~20k iterations
- [ ] Check that DPT weights were successfully copied (initialization logs)

## Troubleshooting Migration Issues

### Issue: "MapAnything model doesn't have dpt_feature_head"
**Cause**: MapAnything checkpoint doesn't have DPT head (only has linear head)

**Solution**: Weight initialization will fall back to random. Still works but loses geometric prior benefit.

### Issue: OOM (Out of Memory) errors
**Cause**: Panoptic DPT has more trainable parameters

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 2`
2. Use gradient checkpointing (requires code modification)
3. Reduce image size: `cfg.INPUT.IMAGE_SIZE = 800`

### Issue: Different loss values compared to working script
**Cause**: Different channel dimensions affect loss scale

**Solution**: This is expected. Compare PQ metrics instead of raw loss values.

### Issue: Slower training speed
**Cause**: More parameters to update

**Solution**: This is expected. Each iteration trains 5-6x more parameters.

## Expected Improvements

Based on architectural changes, expect:

1. **Better initial features**: DPT starts with geometric knowledge
2. **Faster convergence**: Pretrained weights reduce random exploration
3. **Better semantic alignment**: DPT learns geometric→semantic mapping
4. **Improved PQ**: Proper multi-scale features help both classification and segmentation

## Backwards Compatibility

The two scripts are **NOT** compatible:
- Different backbone architectures
- Different checkpoint formats
- Different configuration parameters

**Cannot** resume Panoptic DPT training from working script checkpoint or vice versa.

## Recommended Workflow

1. **Start with verification**: `python verify_panoptic_dpt.py`
2. **Short test run**: Train for 1000 iterations, verify no errors
3. **Compare with working script**: Run both for 10k iterations, compare PQ
4. **Full training**: Once validated, run full 90k iterations
5. **Ensemble (optional)**: Keep both models, ensemble predictions

## Code Reusability

Components that work with both scripts:
- ✅ COCO dataset registration
- ✅ Evaluation code
- ✅ Visualization tools
- ✅ Configuration for Mask2Former head
- ❌ Backbone definition (completely different)
- ❌ Optimizer setup (different parameter groups)

## Summary

The Panoptic DPT pipeline represents a more principled approach:
- **Geometric → Semantic transfer** instead of random initialization
- **Proper multi-scale pyramid** instead of uniform channels  
- **Trainable adaptation layer** instead of frozen DPT
- **Differential learning** to preserve geometric knowledge while learning semantic patterns

Trade-off: More complexity and memory for potentially better performance.
