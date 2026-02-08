# ScanNet++ Multi-View Training Setup Guide

## Quick Start

This guide helps you configure and run multi-view panoptic segmentation training on ScanNet++ using MapAnything + Mask2Former with Query Propagation.

---

## 1. Required Directory Structure

Your ScanNet++ data should be organized as follows:

```
/path/to/scannetpp/
├── data/                              # Main data directory (SCANNETPP_ROOT)
│   ├── <scene_id>/                    # e.g., 0a5c013435
│   │   ├── scans/
│   │   │   ├── mesh_aligned_0.05.ply  # 3D mesh (required for depth rendering)
│   │   │   ├── mesh_aligned_0.05_semantic.ply  # Semantic mesh (for label generation)
│   │   │   ├── segments.json          # Segment IDs per vertex
│   │   │   └── segments_anno.json     # Instance annotations
│   │   ├── dslr/
│   │   │   ├── resized_images/        # Original fisheye images
│   │   │   ├── resized_undistorted_images/  # Undistorted images (recommended)
│   │   │   └── nerfstudio/
│   │   │       ├── transforms.json
│   │   │       └── transforms_undistorted.json  # Camera poses
│   │   └── iphone/                    # (Optional) iPhone captures
│   │       ├── rgb/
│   │       └── depth/                 # LiDAR depth
│   ├── <scene_id_2>/
│   └── ...
│
├── splits/                            # Dataset splits
│   ├── nvs_sem_train.txt              # Training scene IDs
│   ├── nvs_sem_val.txt                # Validation scene IDs
│   └── nvs_sem_test.txt               # Test scene IDs
│
└── panoptic_annotations/              # PANOPTIC_ROOT (auto-generated or manual)
    ├── <scene_id>/
    │   ├── <image_name>.png           # Panoptic annotation per image
    │   ├── <image_name>.json          # Segments info per image
    │   └── ...
    └── ...
```

---

## 2. Panoptic Label Generation (NEW!)

ScanNet++ provides 3D mesh annotations, not 2D panoptic labels. The training script can **automatically generate** 2D labels from the 3D mesh.

### Option A: Generate labels before training (recommended)

```bash
# Generate labels only (no training)
python m2f_train_multiview.py \
    --generate-labels-only \
    --scannetpp-root /path/to/scannetpp/data \
    --panoptic-root /path/to/scannetpp/panoptic_annotations \
    --split-dir /path/to/scannetpp/splits \
    --label-gen-workers 8
```

### Option B: Generate labels and train in one command

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --generate-labels \
    --scannetpp-root /path/to/scannetpp/data \
    --panoptic-root /path/to/scannetpp/panoptic_annotations \
    --num-gpus 4
```

### Option C: Use standalone script

```bash
python tools/generate_panoptic_labels.py \
    --scannetpp-root /path/to/scannetpp/data \
    --output-dir /path/to/panoptic_annotations \
    --split-file /path/to/splits/nvs_sem_train.txt \
    --num-workers 8
```

### Label Generation Output

For each image, the script generates:
- `<image_name>.png`: Panoptic segmentation (RGB encoded)
- `<image_name>.json`: Segments info with category IDs and areas

**Encoding**: `panoptic_id = R + G*256 + B*65536`
- `semantic_id = panoptic_id // 10000`
- `instance_id = panoptic_id % 10000`
- Uses 10000 multiplier to support ScanNet++ 1000 semantic classes

---

## 3. Configuration Values You MUST Set

### Option A: Set via command line arguments

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --num-gpus 4 \
    --scannetpp-root /path/to/scannetpp/data \
    --panoptic-root /path/to/scannetpp/panoptic_annotations \
    OUTPUT_DIR ./output/my_experiment
```

### Option B: Create a local config override

Create a file `configs/scannetpp/panoptic-segmentation/my_training.yaml`:

```yaml
_BASE_: multiview_mapanything.yaml

# Your specific dataset paths are set in the training script arguments
# This file is for hyperparameter overrides

SOLVER:
  IMS_PER_BATCH: 4       # Adjust based on GPU memory
  MAX_ITER: 90000

MODEL:
  MULTIVIEW:
    NUM_VIEWS: 4         # 2, 4, or 8 views per scene
  SEM_SEG_HEAD:
    NUM_CLASSES: 1000    # ScanNet++ has 1000 semantic classes

OUTPUT_DIR: "./output/scannetpp_experiment"
```

---

## 4. Dataset Registration

Dataset registration is now **automatic** when you provide `--scannetpp-root` and `--panoptic-root`.

The script registers:
- `scannetpp_panoptic_train`: Training dataset
- `scannetpp_panoptic_val`: Validation dataset

---

## 5. Required Dependencies

```bash
# Core dependencies
pip install torch torchvision  # PyTorch 2.0+
pip install detectron2         # Detectron2

# MapAnything (required for backbone)
pip install mapanything
# OR install from source:
# pip install git+https://github.com/nkeetha/map-anything.git

# For depth rendering from mesh
pip install open3d              # GPU-accelerated raycasting

# Other dependencies
pip install scipy numpy opencv-python
pip install pycocotools        # For panoptic evaluation
```

---

## 6. Pre-trained Model

Ensure MapAnything model is available:

```bash
# Option 1: Auto-download (default)
# The model downloads automatically from HuggingFace on first run

# Option 2: Manual download
mkdir -p pretrained_models/map_anything
# Download from: https://huggingface.co/nkeetha/map-anything
# Place model.pt and config.yaml in pretrained_models/map_anything/
```

---

## 7. Training Commands

### Full Pipeline (Generate Labels + Train)

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --generate-labels \
    --num-gpus 4 \
    --scannetpp-root /path/to/scannetpp/data \
    --panoptic-root /path/to/scannetpp/panoptic_annotations \
    --split-dir /path/to/scannetpp/splits \
    OUTPUT_DIR ./output/scannetpp_multiview
```

### Basic Training (4 GPUs)

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --num-gpus 4 \
    OUTPUT_DIR ./output/scannetpp_multiview_4gpu
```

### Single GPU Training (adjust batch size)

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 1 \
    SOLVER.BASE_LR 0.000025 \
    OUTPUT_DIR ./output/scannetpp_multiview_1gpu
```

### With Custom Paths

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --num-gpus 4 \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 90000 \
    MODEL.MULTIVIEW.NUM_VIEWS 4 \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES 1000 \
    INPUT.LOAD_DEPTH True \
    INPUT.DEPTH_CACHE_DIR ./depth_cache \
    OUTPUT_DIR ./output/experiment_v1
```

### Resume Training

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --resume \
    MODEL.WEIGHTS ./output/experiment_v1/model_0049999.pth
```

---

## 8. View Analysis Mode

After training, run detailed per-view analysis:

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview_mapanything.yaml \
    --eval-only \
    --view-analysis \
    MODEL.WEIGHTS ./output/experiment_v1/model_final.pth \
    OUTPUT_DIR ./output/view_analysis
```

This generates `view_analysis_metrics.csv` with per-view loss breakdown.

---

## 8. Configuration Reference

### Key Parameters to Tune

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL.MULTIVIEW.NUM_VIEWS` | 4 | Views per scene (2, 4, or 8) |
| `MODEL.MULTIVIEW.WARMUP_ITER` | 1000 | Iters before query propagation |
| `MODEL.MULTIVIEW.MIN_CAMERA_DISTANCE` | 0.3 | Min distance between views (m) |
| `MODEL.SEM_SEG_HEAD.NUM_CLASSES` | 1000 | Number of semantic classes |
| `SOLVER.IMS_PER_BATCH` | 4 | Scenes per batch |
| `SOLVER.BASE_LR` | 0.0001 | Learning rate |
| `SOLVER.MAX_ITER` | 90000 | Training iterations |
| `INPUT.LOAD_DEPTH` | True | Enable depth for spatial bridging |
| `INPUT.IMAGE_SIZE` | 640 | Input image resolution |

### Memory Optimization

If you run out of GPU memory:

1. **Reduce batch size**: `SOLVER.IMS_PER_BATCH 2`
2. **Reduce views**: `MODEL.MULTIVIEW.NUM_VIEWS 2`
3. **Reduce image size**: `INPUT.IMAGE_SIZE 480`
4. **Enable gradient checkpointing** (requires code modification)

---

## 9. Expected Output Structure

After training:

```
output/scannetpp_multiview/
├── model_final.pth           # Final model weights
├── model_0004999.pth         # Checkpoint at iter 5000
├── model_0009999.pth         # Checkpoint at iter 10000
├── ...
├── metrics.json              # Detectron2 metrics
├── training_metrics.csv      # Per-iteration training losses
├── evaluation_metrics.csv    # PQ/SQ/RQ at each eval period
├── view_analysis_metrics.csv # Per-view breakdown (if --view-analysis)
└── events.out.tfevents.*     # TensorBoard logs
```

---

## 10. Common Issues & Solutions

### Issue: "Dataset not found"
**Solution**: Register the dataset before training. See Section 3.

### Issue: "CUDA out of memory"
**Solution**: Reduce `SOLVER.IMS_PER_BATCH` or `MODEL.MULTIVIEW.NUM_VIEWS`

### Issue: "Mesh file not found" (depth rendering)
**Solution**: Ensure `mesh_aligned_0.05.ply` exists in each scene's `scans/` folder

### Issue: "Camera poses not found"
**Solution**: Verify `transforms_undistorted.json` exists in `dslr/nerfstudio/`

### Issue: Training loss is NaN
**Solution**: The NaNLossCheckHook will catch this. Try:
- Lower learning rate: `SOLVER.BASE_LR 0.00005`
- Enable gradient clipping (already enabled by default)

---

## 11. Panoptic Annotation Format

Your panoptic annotations (`panoptic_annotations/<scene_id>/<image>.png`) should be:

- **Format**: PNG with 3 channels
- **Encoding**: `panoptic_id = R + G*256 + B*256*256`
- **Structure**: `panoptic_id = semantic_id * 10000 + instance_id`
  - `semantic_id = panoptic_id // 10000`
  - `instance_id = panoptic_id % 10000`
  - Uses 10000 multiplier to support ScanNet++ 1000 semantic classes
- **Void/ignore**: Use value 0 or 255 for unlabeled regions

If you need to generate these from ScanNet++ 3D annotations, you'll need a rasterization script that projects 3D labels to 2D.

---

## Quick Checklist Before Training

- [ ] ScanNet++ data downloaded and organized per Section 1
- [ ] Panoptic annotations rasterized to 2D PNGs
- [ ] Split files exist (`nvs_sem_train.txt`, `nvs_sem_val.txt`)
- [ ] MapAnything model accessible (auto-downloads or pre-placed)
- [ ] Dependencies installed (PyTorch, Detectron2, Open3D, etc.)
- [ ] Dataset registered in training script
- [ ] Config file paths verified
- [ ] Output directory writable
- [ ] GPU memory sufficient for batch size

---

## Contact

For issues with:
- **This training code**: Check the repository issues
- **ScanNet++ dataset**: https://kaldir.vc.in.tum.de/scannetpp/
- **MapAnything**: https://github.com/nkeetha/map-anything
- **Mask2Former**: https://github.com/facebookresearch/Mask2Former
