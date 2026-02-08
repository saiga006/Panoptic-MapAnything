# Per-View Loss and Class Probability Analysis

## Overview

Added detailed per-view analysis capabilities to track reference vs target view losses and class probability consistency across views. This helps understand how well query propagation maintains prediction consistency across different viewpoints.

## New Features

### 1. Detailed View Metrics Computation

The `compute_view_metrics()` method in `MultiViewMask2Former` computes:

- **Per-view losses**: Separate loss tracking for reference view and each target view
  - Classification loss (loss_ce)
  - Mask loss (loss_mask)
  - Dice loss (loss_dice)
  
- **Class probability consistency**: Measures how consistent class predictions are across views
  - KL divergence between reference and target view predictions
  - Top-class agreement (percentage of queries with same top class)

### 2. CSV Logging

Three CSV files are generated in the output directory:

#### `training_metrics.csv`
Logs training losses every N iterations (default: 20)
```csv
iteration,total_loss,loss_ce,loss_mask,loss_dice,lr
100,2.341234,0.567890,1.234567,0.456789,0.00010000
```

#### `evaluation_metrics.csv`
Logs panoptic quality metrics after each evaluation period
```csv
iteration,PQ,SQ,RQ,PQ_th,SQ_th,RQ_th,PQ_st,SQ_st,RQ_st
5000,45.23,78.45,56.78,42.12,76.34,54.23,48.90,80.12,59.45
```

#### `view_analysis_metrics.csv`
Logs detailed per-view analysis (when view analysis mode is enabled)
```csv
iteration,scene_id,view_type,view_idx,loss_ce,loss_mask,loss_dice,total_loss,class_prob_kl,top_class_agreement
0,0a5c013435,ref,0,0.234567,0.456789,0.123456,0.814812,0.0,1.0
0,0a5c013435,target,1,0.245678,0.467890,0.134567,0.848135,0.034521,0.892345
0,0a5c013435,target,2,0.256789,0.478901,0.145678,0.881368,0.045632,0.876543
```

### 3. View Analysis Mode

Run detailed view analysis on evaluation dataset:

```bash
python m2f_train_multiview.py \
    --eval-only \
    --view-analysis \
    --config-file configs/scannetpp/my_config.yaml \
    MODEL.WEIGHTS output/model_final.pth
```

This will:
- Process each scene in the evaluation dataset
- Compute per-view losses and class probability metrics
- Log results to `view_analysis_metrics.csv`
- Print progress summaries every 10 scenes

You can control the number of scenes analyzed with:
```python
cfg.TEST.VIEW_ANALYSIS_MAX_SAMPLES = 100  # Default: 100 scenes
```

## Understanding the Metrics

### Reference vs Target Losses

- **Reference view**: The view used to initialize queries (typically view 0)
- **Target views**: Other views that receive propagated queries

Compare reference vs target losses to understand:
- How well queries transfer across views
- Which views are harder to segment (higher losses)
- Whether certain viewpoints are consistently problematic

### Class Probability Consistency

#### KL Divergence (`class_prob_kl`)
- Measures how different the class probability distributions are
- **Lower is better**: 0 = identical distributions
- High KL divergence indicates the model predicts different classes for the same query across views

#### Top-Class Agreement (`top_class_agreement`)
- Percentage of queries where the top predicted class matches the reference view
- **Higher is better**: 1.0 = perfect agreement
- Low agreement suggests inconsistent predictions across views

## Usage Examples

### Training with Automatic Logging

The CSV loggers are automatically registered during training:

```python
trainer = MultiViewTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

This will automatically create:
- `training_metrics.csv` - updated every 20 iterations
- `evaluation_metrics.csv` - updated after each evaluation

### Analyzing View Consistency

After training, run view analysis:

```bash
python m2f_train_multiview.py \
    --eval-only \
    --view-analysis \
    --config-file configs/scannetpp/panoptic_multiview.yaml \
    MODEL.WEIGHTS output/model_0004999.pth
```

### Plotting Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load view analysis results
df = pd.read_csv('output/view_analysis_metrics.csv')

# Plot per-view loss breakdown
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Reference vs target losses
ref_df = df[df['view_type'] == 'ref']
target_df = df[df['view_type'] == 'target']

axes[0].hist(ref_df['loss_ce'], alpha=0.5, label='Reference', bins=30)
axes[0].hist(target_df['loss_ce'], alpha=0.5, label='Target', bins=30)
axes[0].set_xlabel('Classification Loss')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# KL divergence distribution
axes[1].hist(target_df['class_prob_kl'], bins=30)
axes[1].set_xlabel('KL Divergence (from reference)')
axes[1].set_ylabel('Frequency')

# Top-class agreement
axes[2].hist(target_df['top_class_agreement'], bins=30)
axes[2].set_xlabel('Top-Class Agreement')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('view_analysis.png')
```

## Implementation Details

### Model Method: `compute_view_metrics()`

Located in `MultiViewMask2Former` class:

1. Runs backbone on all views
2. Processes reference view to get initial queries
3. Processes each target view with propagated queries
4. Computes losses for each view separately
5. Calculates class probability consistency metrics

### CSV Logger: `CSVMetricsLogger`

A Detectron2 `HookBase` that:
- Initializes CSV files with headers
- Logs training metrics in `after_step()` hook
- Logs evaluation metrics in `after_eval()` hook
- Provides `log_view_analysis()` method for per-view data

### Trainer: `MultiViewTrainer.run_view_analysis()`

Static method that:
- Builds evaluation data loader
- Iterates through scenes
- Calls `model.compute_view_metrics()` for each scene
- Logs results to CSV
- Prints progress summaries

## Configuration

Add to your config file:

```python
# Control how many samples to analyze
cfg.TEST.VIEW_ANALYSIS_MAX_SAMPLES = 100

# Logging frequency for training metrics (iterations)
cfg.SOLVER.LOGGING_PERIOD = 20  # Default in Detectron2
```

## Expected Behavior

### Good Query Propagation
- Target view losses similar to reference losses
- Low KL divergence (< 0.1)
- High top-class agreement (> 0.85)

### Poor Query Propagation
- Target view losses significantly higher than reference
- High KL divergence (> 0.5)
- Low top-class agreement (< 0.60)

This can indicate:
- Insufficient geometric warping quality
- Large viewpoint changes breaking feature correspondence
- Need for more training data from diverse viewpoints
- Depth estimation errors affecting attention mask warping
