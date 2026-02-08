# ScanNet++ Integration for MapAnything + Mask2Former

This document explains how to use the ScanNet++ dataset with the MapAnything + Mask2Former panoptic segmentation architecture.

## Table of Contents

1. [ScanNet++ Dataset Structure](#scannet-dataset-structure)
2. [Dataset Mapper Design](#dataset-mapper-design)
3. [View Selection Strategy](#view-selection-strategy)
4. [Training Approaches](#training-approaches)
5. [Multi-View Adaptation](#multi-view-adaptation)
6. [3D Consistency Training (Dual-DPT)](#3d-consistency-training-dual-dpt)
7. [Usage Examples](#usage-examples)

---

## ScanNet++ Dataset Structure

ScanNet++ is a high-fidelity indoor 3D dataset with both DSLR and iPhone captures. Here's the complete folder structure:

```
scannetpp/
├── splits/
│   ├── nvs_sem_train.txt      # Training scenes (856 scenes)
│   ├── nvs_sem_val.txt        # Validation scenes (50 scenes)
│   ├── nvs_test.txt           # NVS test scenes (50 scenes)
│   └── sem_test.txt           # Semantic test scenes (50 scenes)
├── metadata/
│   ├── scene_types.json       # Scene type mapping
│   ├── semantic_classes.txt   # 1500+ semantic classes
│   ├── instance_classes.txt   # Instance-level classes
│   └── semantic_benchmark/
│       ├── top100.txt         # Top 100 benchmark classes
│       └── map_benchmark.csv  # Raw to benchmark mapping
└── data/
    └── <scene_id>/
        ├── scans/
        │   ├── mesh_aligned_0.05.ply           # 3D mesh (5% decimated)
        │   ├── mesh_aligned_0.05_semantic.ply  # Mesh with semantic labels
        │   ├── segments.json                    # Segment IDs per vertex
        │   ├── segments_anno.json               # Instance annotations
        │   ├── scanner_poses.json               # Scanner 4x4 transforms
        │   ├── pc_aligned.ply                   # Point cloud
        │   └── pc_aligned_mask.txt              # Anonymization mask
        ├── dslr/
        │   ├── resized_images/                  # Fisheye DSLR images (2MP)
        │   ├── resized_anon_masks/              # Anonymization masks
        │   ├── original_images/                 # Full resolution (33MP)
        │   ├── resized_undistorted_images/      # Undistorted pinhole
        │   ├── resized_undistorted_masks/
        │   ├── colmap/
        │   │   ├── cameras.txt      # OPENCV_FISHEYE intrinsics
        │   │   ├── images.txt       # Extrinsics (qvec, tvec)
        │   │   └── points3D.txt     # 3D feature points
        │   └── nerfstudio/
        │       ├── transforms.json              # OpenGL convention poses
        │       └── transforms_undistorted.json  # For undistorted images
        ├── iphone/
        │   ├── rgb.mkv              # Full RGB video (60fps)
        │   ├── rgb_mask.mkv         # Anonymization mask video
        │   ├── depth.bin            # LiDAR depth binary
        │   ├── rgb/                 # Extracted frames (1920x1440)
        │   ├── depth/               # Depth maps (256x192, 16-bit PNG mm)
        │   ├── pose_intrinsic_imu.json  # ARKit poses + intrinsics
        │   ├── colmap/              # OPENCV intrinsics
        │   └── nerfstudio/
        │       └── transforms.json
        └── panocam/                 # Panoramic scanner images
            ├── images/
            ├── depth/
            └── ...
```

### Camera Data Formats

#### COLMAP Format (world-to-camera)
```
# cameras.txt
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 OPENCV_FISHEYE 1920 1440 fx fy cx cy k1 k2 k3 k4

# images.txt
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
1 0.9 0.1 0.2 0.3 1.0 2.0 3.0 1 DSC00001.JPG
```

Quaternion format: `(w, x, y, z)`  
Transform: `x_cam = R @ x_world + t`

#### NerfStudio Format (camera-to-world)
```json
{
    "fl_x": 1000.0,
    "fl_y": 1000.0,
    "cx": 960.0,
    "cy": 720.0,
    "w": 1920,
    "h": 1440,
    "camera_model": "OPENCV_FISHEYE",
    "k1": -0.1, "k2": 0.05, "k3": 0.0, "k4": 0.0,
    "frames": [
        {
            "file_path": "./resized_images/DSC00001.JPG",
            "transform_matrix": [[...4x4 matrix...]],
            "mask_path": "./resized_anon_masks/DSC00001.png",
            "is_bad": false
        }
    ],
    "test_frames": [...]
}
```

**Convention**: NerfStudio uses OpenGL/Blender convention:
- +X right, +Y up, -Z forward
- Must convert to OpenCV (+X right, +Y down, +Z forward) for most vision tasks

---

## Dataset Mapper Design

### ScanNetPPPanopticDatasetMapper (Single-View)

This mapper handles single images with camera poses:

```python
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    ScanNetPPPanopticDatasetMapper
)

# Input dataset_dict format:
{
    'file_name': '/path/to/image.jpg',
    'scene_id': 'abc123',
    'image_id': 0,
    'height': 1440,
    'width': 1920,
    'camera_to_world': np.ndarray([4, 4]),  # 4x4 pose matrix
    'intrinsics': {
        'fl_x': 1000.0, 'fl_y': 1000.0,
        'cx': 960.0, 'cy': 720.0,
        'k1': 0.0, 'k2': 0.0, ...
    },
    'pan_seg_file_name': '/path/to/panoptic.png',  # Optional
    'segments_info': [...]                          # Optional
}

# Output format:
{
    'image': torch.Tensor([3, H, W]),
    'camera_to_world': torch.Tensor([4, 4]),
    'intrinsic_matrix': torch.Tensor([3, 3]),
    'instances': Instances,  # Panoptic masks
    ...
}
```

### ScanNetPPMultiViewDatasetMapper (Multi-View)

For MapAnything's multi-view fusion capability:

```python
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    ScanNetPPMultiViewDatasetMapper
)

# Input: just scene_id, mapper handles view selection
{
    'scene_id': 'abc123'
}

# Output: batched views
{
    'images': torch.Tensor([N, 3, H, W]),        # N views stacked
    'camera_to_world': torch.Tensor([N, 4, 4]),  # Pose per view
    'intrinsics': torch.Tensor([4, 4]),          # Shared intrinsics
    'num_views': N,
    'selected_view_indices': [int, ...],
}
```

---

## View Selection Strategy

### Problem: Overlapping Views

Indoor scenes often have many images from similar viewpoints. Training on highly overlapping views:
- Wastes compute on redundant information
- Reduces diversity of training samples
- Can lead to overfitting to specific viewpoints

### Solution: Farthest Point Sampling

We implement **farthest point sampling** for view selection:

```python
def select_diverse_views(poses, num_views, min_distance):
    """
    Algorithm:
    1. Start with a random view
    2. Iteratively add the view that maximizes minimum distance 
       to all already-selected views
    3. Stop when num_views reached or no view satisfies min_distance
    """
```

### View Overlap Metrics

We compute overlap between views using:

1. **Camera Distance**: Euclidean distance between camera centers
   ```python
   distance = ||center1 - center2||
   ```

2. **Viewing Direction Similarity**: Cosine of angle between viewing directions
   ```python
   similarity = dot(dir1, dir2)  # -1 to 1
   ```

3. **Combined Overlap Score**:
   ```python
   overlap = distance_score * direction_score
   # where distance_score = max(0, 1 - distance/threshold)
   # and direction_score = (similarity + 1) / 2
   ```

### Configuration

```python
# In config or command line:
cfg.INPUT.NUM_VIEWS = 4          # Views per sample
cfg.INPUT.MIN_VIEW_DISTANCE = 0.5  # Minimum 50cm between views
```

---

## Training Approaches

### 1. Single Scene Training (Debugging/Overfitting Test)

Train on a single scene to verify the pipeline:

```bash
python m2f_train_scannetpp.py \
    --mapanything-checkpoint /path/to/mapanything.pt \
    --scannetpp-root /path/to/scannetpp/data \
    --scene-id abc123 \
    --num-train-views 50 \
    --num-val-views 10 \
    --min-view-distance 0.3 \
    --max-iter 5000 \
    --output-dir ./output_single_scene
```

**Use cases:**
- Verify data loading works correctly
- Debug model architecture
- Quick iteration during development
- Scene-specific fine-tuning

### 2. Multi-Scene Training

Train across all scenes in a split:

```bash
python m2f_train_scannetpp.py \
    --mapanything-checkpoint /path/to/mapanything.pt \
    --scannetpp-root /path/to/scannetpp/data \
    --split-file /path/to/splits/nvs_sem_train.txt \
    --panoptic-dir /path/to/panoptic_annotations \
    --num-gpus 4 \
    --batch-size 2 \
    --max-iter 90000 \
    --output-dir ./output_full_training
```

### 3. Multi-View Training (MapAnything Fusion)

Enable MapAnything's multi-view transformer:

```bash
python m2f_train_scannetpp.py \
    --mapanything-checkpoint /path/to/mapanything.pt \
    --scannetpp-root /path/to/scannetpp/data \
    --split-file /path/to/splits/nvs_sem_train.txt \
    --num-views 4 \
    --min-view-distance 0.5 \
    --num-gpus 4 \
    --output-dir ./output_multiview
```

---

## Multi-View Adaptation

### Current Architecture (Single-View)

```
Input Image [B, 3, H, W]
    ↓
┌─────────────────────────────────────────┐
│  FROZEN: MapAnything Backbone           │
│  - DINOv2 Encoder                       │
│  - Multi-View Transformer (N=1)         │ ← Currently single view
│  Output: 4 layer features               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Panoptic DPT Head           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Mask2Former Head            │
│  Output: Panoptic Predictions           │
└─────────────────────────────────────────┘
```

### Multi-View Architecture (Implemented)

```
Input Views [B, N, 3, H, W] + Poses [B, N, 4, 4]
    ↓
┌─────────────────────────────────────────┐
│  FROZEN: MapAnything Backbone           │
│  - DINOv2 Encoder (per view)            │
│  - Multi-View Transformer (N views)     │ ← Full multi-view fusion!
│  - Cross-view attention with poses      │
│  Output: Fused 4 layer features         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Panoptic DPT Head           │
│  - Receives fused multi-view features   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TRAINABLE: Mask2Former Head            │
│  Output: Panoptic Predictions           │
└─────────────────────────────────────────┘
```

### Multi-View Implementation Details

The multi-view training is implemented in `m2f_train_multiview.py`. Here's how it works:

#### 1. Camera Pose Format (MapAnything expects)

```python
# Each view dictionary contains:
view = {
    'img': tensor,                    # [B, 3, H, W]
    'data_norm_type': ['dinov2'],     # Encoder normalization
    'camera_pose_quats': tensor,      # [B, 4] quaternion (x, y, z, w)
    'camera_pose_trans': tensor,      # [B, 3] translation
}

# Poses are camera-to-world transforms
# MapAnything internally converts to relative poses (view N relative to view 0)
```

#### 2. Multi-View Forward Pass

```python
# In MapAnythingMultiViewBackbone.forward():

# Step 1: Encode all N views through DINOv2
all_encoder_features = self.mapanything._encode_n_views(views)
# Returns: tuple of N tensors, each [B, C, patch_h, patch_w]

# Step 2: Optional geometric fusion (camera poses → tokens)
if self.use_poses and camera_poses is not None:
    all_encoder_features = self.mapanything._encode_and_fuse_optional_geometric_inputs(
        views, all_encoder_features
    )
# This encodes camera rotations and translations as additional tokens

# Step 3: Multi-view attention (cross-attention between views)
info_sharing_input = MultiViewTransformerInput(
    features=list(all_encoder_features),  # List of N tensors
    additional_input_tokens=input_scale_token,
)
info_sharing_output = self.mapanything.info_sharing(info_sharing_input)
# The transformer performs cross-attention between all N views
```

#### 3. MultiViewTransformerInput Structure

From the uniception source code:
```python
@dataclass
class MultiViewTransformerInput(InfoSharingInput):
    # List of feature tensors, one per view
    features: List[Float[Tensor, "batch input_embed_dim feat_height feat_width"]]
    
    # Optional additional tokens (e.g., scale token)
    additional_input_tokens: Optional[Float[Tensor, "batch input_embed_dim num_tokens"]] = None
```

#### 4. Data Collation for Multi-View

```python
# Custom collate function packs multiple views:
def multi_view_collate_fn(batch_list):
    return {
        'images': tensor,          # [B, N, 3, H, W]
        'camera_poses': tensor,    # [B, N, 4, 4]
        'camera_intrinsics': tensor,  # [B, N, 3, 3]
        'sem_seg': tensor,         # [B, N, H, W]
        'instances': list,         # [B * N] Instances objects
        'scene_ids': list,         # [B] scene IDs
    }
```

#### 5. View Selection (Minimal Overlap)

Implemented using farthest point sampling:
```python
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    select_views_for_scene,
)

# Select N diverse views with minimum distance between cameras
selected_indices = select_views_for_scene(
    frames,               # List of frame dicts with 'camera_to_world'
    num_views=4,          # Target number of views
    min_distance=0.5,     # Minimum 50cm between cameras
    seed=42,              # For reproducibility
)
```

---

## 3D Consistency Training (Dual-DPT)

This section describes the 3D consistency training approach implemented in `m2f_train_3d_consistency.py`. This approach is inspired by the UNITE paper but adapted for the Mask2Former query-based paradigm.

### Key Insight: Dual-DPT Strategy

MapAnything has **two DPT heads** with different purposes:

1. **Geometric DPT (frozen)**: Outputs depth and confidence maps
   - Used to compute per-pixel reliability
   - Provides confidence weights for loss computation
   
2. **Panoptic DPT (trainable)**: Outputs dense features for segmentation
   - These are the features we want to align across views
   - Connected to Mask2Former pixel decoder

### Architecture Overview

```
Input Views [B, N, 3, H, W] + Poses [B, N, 4, 4]
    ↓
┌─────────────────────────────────────────────────┐
│  FROZEN: MapAnything Core                       │
│  - DINOv2 Encoder (per view)                    │
│  - Multi-View Transformer (cross-attention)     │
└─────────────────────────────────────────────────┘
    ↓ shared encoder features
    ├──────────────────────────────────────────────┐
    ↓                                              ↓
┌─────────────────────────┐    ┌─────────────────────────────────┐
│  FROZEN: Geometric DPT  │    │  TRAINABLE: Panoptic DPT        │
│  Output:                │    │  Output:                        │
│  - Depth [B*N, 1, H, W] │    │  - res2 [B*N, C2, H/4, W/4]     │
│  - Confidence [B*N,1,H,W│    │  - res3 [B*N, C3, H/8, W/8]     │
└─────────────────────────┘    │  - res4 [B*N, C4, H/16, W/16]   │
         ↓                     │  - res5 [B*N, C5, H/32, W/32]   │
         ↓                     └─────────────────────────────────┘
         ↓                                  ↓
    Confidence weights         ┌─────────────────────────────────┐
         ↓                     │  TRAINABLE: Mask2Former         │
         ↓                     │  - MSDeformAttn Pixel Decoder   │
         ↓                     │  - Transformer Decoder          │
         ↓                     │  Output:                        │
         ↓                     │  - Query embeddings [B*N, Q, D] │
         ↓                     │  - Mask predictions [B*N, Q,H,W]│
         ↓                     └─────────────────────────────────┘
         ↓                                  ↓
    ┌────────────────────────────────────────┐
    │      3D CONSISTENCY LOSSES             │
    │                                        │
    │  1. Dense Feature Consistency          │
    │     - Align Panoptic DPT features      │
    │     - Weight by Geometric DPT conf     │
    │                                        │
    │  2. Query Contrastive Loss             │
    │     - Group queries by 3D location     │
    │     - Pull same-point queries together │
    │                                        │
    │  3. Mask Projection Consistency        │
    │     - Lift masks to 3D using depth     │
    │     - Reproject to other views         │
    │     - Penalize inconsistencies         │
    └────────────────────────────────────────┘
```

### 3D Consistency Losses

#### 1. Dense Feature Consistency Loss

Aligns Panoptic DPT features across views using pixel correspondences:

```python
# For each pair of views (i, j):
# 1. Unproject view_i pixels to 3D using Geometric DPT depth
# 2. Project 3D points to view_j
# 3. Compare features at corresponding pixels
# 4. Weight by confidence from both views

def compute_dense_feature_consistency(
    panoptic_features,    # From Panoptic DPT (trainable)
    depths,               # From Geometric DPT (frozen)
    confidences,          # From Geometric DPT (frozen)
    camera_poses,
    intrinsics,
):
    # Find correspondences via depth-based reprojection
    corresp = compute_correspondences(depths, camera_poses, intrinsics)
    
    # Extract features at corresponding points
    feat_i = sample_features(panoptic_features[i], corresp.pts_i)
    feat_j = sample_features(panoptic_features[j], corresp.pts_j)
    
    # Weight by confidence (from Geometric DPT)
    weights = confidences[i] * confidences[j]
    
    # L2 loss with confidence weighting
    loss = (weights * (feat_i - feat_j).pow(2)).mean()
```

#### 2. Query Contrastive Loss

Groups Mask2Former queries by their 3D location:

```python
def compute_query_consistency(
    query_embeddings,     # [N, Q, D] from Mask2Former
    mask_predictions,     # [N, Q, H, W]
    depths,
    camera_poses,
    intrinsics,
    temperature=0.1,
):
    # For each query, compute 3D centroid of its mask
    query_3d_positions = []
    for view_idx in range(N):
        for q in range(Q):
            mask = mask_predictions[view_idx, q]  # [H, W]
            # Get 3D positions of mask pixels
            pts_3d = unproject_mask(mask, depths[view_idx], camera_poses[view_idx])
            centroid = pts_3d.mean(dim=0)  # [3]
            query_3d_positions.append(centroid)
    
    # Group queries by 3D proximity
    # Queries within threshold are positive pairs
    positive_pairs = find_nearby_queries(query_3d_positions, threshold=0.1)
    
    # InfoNCE contrastive loss
    loss = info_nce_loss(query_embeddings, positive_pairs, temperature)
```

#### 3. Mask Projection Consistency

Ensures masks are consistent when projected to 3D:

```python
def compute_mask_projection_consistency(
    mask_predictions,     # [N, Q, H, W]
    depths,
    confidences,
    camera_poses,
    intrinsics,
):
    loss = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            
            # Lift mask from view_i to 3D
            mask_3d = lift_mask_to_3d(
                mask_predictions[i],
                depths[i],
                camera_poses[i],
                intrinsics[i]
            )
            
            # Project to view_j
            mask_reproj = project_3d_to_view(
                mask_3d,
                camera_poses[j],
                intrinsics[j]
            )
            
            # Compare with actual mask in view_j
            # Weight by confidence at projected pixels
            weights = sample_confidence(confidences[j], mask_reproj.coords)
            loss += (weights * (mask_reproj - mask_predictions[j]).abs()).mean()
```

### Training Strategy

The training proceeds in stages:

```python
# Stage 1: Standard 2D supervision (warmup)
# - Train Panoptic DPT + Mask2Former on 2D panoptic labels
# - No 3D consistency losses yet

# Stage 2: Add 3D consistency
# - Continue 2D supervision
# - Add confidence-weighted feature consistency
# - Low weight initially, gradually increase

# Stage 3: Full training
# - All losses enabled:
#   - 2D panoptic (ce_loss + dice_loss + mask_loss)
#   - Dense feature consistency
#   - Query contrastive loss
#   - Mask projection consistency

loss_weights = {
    'panoptic_2d': 1.0,           # Always on
    'dense_consistency': 0.1,      # Ramp up from 0
    'query_contrastive': 0.05,     # After warmup
    'mask_projection': 0.1,        # After warmup
}
```

### 3D Consistency Training Command

```bash
python m2f_train_3d_consistency.py \
    --mapanything-checkpoint /path/to/mapanything.pt \
    --scannetpp-root /path/to/scannetpp/data \
    --split-file /path/to/splits/nvs_sem_train.txt \
    --panoptic-dir /path/to/panoptic_2d \
    --num-views 4 \
    --min-view-distance 0.5 \
    --use-3d-consistency \
    --consistency-weight 0.1 \
    --num-gpus 4 \
    --output-dir ./output_3d_consistency
```

### 3D Consistency Configuration

```yaml
MODEL:
  CONSISTENCY:
    ENABLED: True
    DENSE_WEIGHT: 0.1              # Weight for dense feature consistency
    QUERY_WEIGHT: 0.05             # Weight for query contrastive loss  
    MASK_WEIGHT: 0.1               # Weight for mask projection loss
    CONFIDENCE_THRESHOLD: 0.5      # Min confidence to include correspondence
    WARMUP_ITER: 5000              # Iterations before enabling 3D losses
    
  DUAL_DPT:
    GEOMETRIC_HEAD: "geometric"    # Frozen head for depth/confidence
    PANOPTIC_HEAD: "panoptic"      # Trainable head for features
```

### Why This Approach?

The UNITE paper uses dense contrastive learning on 2D feature maps. However, Mask2Former uses a query-based paradigm where:

1. **Queries** are the primary representation (not dense features)
2. **Masks** are predicted per query, not per pixel
3. **Instance identity** is encoded in query embeddings

Our dual-DPT approach bridges these paradigms:

- Use **Geometric DPT confidence** to weight correspondences (like UNITE)
- Align **Panoptic DPT features** for dense consistency (adapted from UNITE)
- Add **query-level contrastive loss** for Mask2Former's paradigm
- Add **mask projection loss** to ensure 3D mask consistency

---

### Multi-View Training Command

```bash
python m2f_train_multiview.py \
    --config-file configs/scannetpp/panoptic-segmentation/multiview.yaml \
    MODEL.MULTIVIEW.NUM_VIEWS 4 \
    MODEL.MULTIVIEW.USE_POSES True \
    MODEL.MULTIVIEW.MIN_CAMERA_DISTANCE 0.5 \
    DATASETS.SCANNETPP_ROOT /path/to/scannetpp/data \
    DATASETS.SCANNETPP_PANOPTIC_ROOT /path/to/panoptic_2d \
    OUTPUT_DIR ./output_multiview
```

### Configuration Options

```yaml
MODEL:
  MULTIVIEW:
    NUM_VIEWS: 4                    # Views per training sample
    USE_POSES: True                 # Use camera poses for fusion
    TRAINING_MODE: "reference_only" # or "all_views"
    MIN_CAMERA_DISTANCE: 0.5        # Minimum distance (meters)
    MAX_CAMERA_DISTANCE: 5.0        # Maximum distance (meters)

DATASETS:
  SCANNETPP_ROOT: "/path/to/scannetpp/data"
  SCANNETPP_PANOPTIC_ROOT: "/path/to/panoptic_2d"
```

---

## Panoptic Label Generation from ScanNet++

ScanNet++ provides **3D mesh-based annotations**, not 2D panoptic labels directly.

### What ScanNet++ Provides

```
scans/
├── mesh_aligned_0.05_semantic.ply  # Per-vertex semantic labels
├── segments.json                    # Segment IDs per vertex
└── segments_anno.json               # Instance annotations
```

### To Generate 2D Panoptic Labels

You need to **render the mesh from each camera viewpoint**:

```python
# Using ScanNet++ toolkit or custom renderer:
# 1. Load mesh with semantic/instance labels
# 2. For each camera pose:
#    - Render mesh using camera intrinsics/extrinsics
#    - Output: sem_seg [H, W] with class IDs
#    - Output: segments_info (list of segment metadata)
```

### Expected 2D Annotation Format

After rasterization, store in:
```
panoptic_2d/
└── <scene_id>/
    ├── DSC00001.png     # Panoptic RGB (encodes segment IDs)
    ├── DSC00001.json    # segments_info
    ├── DSC00002.png
    ├── DSC00002.json
    └── ...
```

**segments_info format:**
```json
[
    {
        "id": 1,           // Unique segment ID (encoded in PNG)
        "category_id": 5,  // Semantic class ID
        "iscrowd": false,
        "isthing": true,   // true=instance, false=stuff
        "area": 12345      // Pixel area
    },
    ...
]
```

---

## Usage Examples

### Example 1: Load Scene Camera Data

```python
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    ScanNetPPScene,
    read_nerfstudio_transforms,
    select_views_for_scene,
)

# Load scene
scene = ScanNetPPScene('abc123', '/path/to/scannetpp/data')

# Read camera data
intrinsics, frames = read_nerfstudio_transforms(
    str(scene.dslr_nerfstudio_transforms_undistorted)
)

# Select diverse views
selected_indices = select_views_for_scene(
    frames,
    num_views=20,
    min_distance=0.5,  # 50cm minimum
    seed=42,
)

print(f"Selected {len(selected_indices)} diverse views")
```

### Example 2: Visualize View Selection

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get camera centers
centers = []
for i, frame in enumerate(frames):
    c2w = frame['camera_to_world']
    center = c2w[:3, 3]
    centers.append(center)

centers = np.array(centers)
selected_centers = centers[selected_indices]

# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# All cameras (gray)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
           c='gray', alpha=0.3, s=10, label='All views')

# Selected cameras (red)
ax.scatter(selected_centers[:, 0], selected_centers[:, 1], selected_centers[:, 2],
           c='red', s=50, label='Selected views')

ax.legend()
plt.title(f'Camera Selection for Scene {scene.scene_id}')
plt.show()
```

### Example 3: Register and Load Dataset

```python
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    register_scannetpp_single_scene,
)
from detectron2.data import DatasetCatalog

# Register single scene
train_name, val_name = register_scannetpp_single_scene(
    scene_id='abc123',
    data_root='/path/to/scannetpp/data',
    num_views_train=50,
    min_view_distance=0.5,
)

# Load dataset
dataset_dicts = DatasetCatalog.get(train_name)
print(f"Loaded {len(dataset_dicts)} training samples")

# Check first sample
sample = dataset_dicts[0]
print(f"Image: {sample['file_name']}")
print(f"Scene: {sample['scene_id']}")
print(f"Pose shape: {sample['camera_to_world'].shape}")
```

---

## Troubleshooting

### Common Issues

1. **Missing undistorted images**:
   ```bash
   # Run ScanNet++ undistortion script first
   python -m dslr.undistort dslr/configs/undistort.yml
   ```

2. **Camera model mismatch**:
   - DSLR uses `OPENCV_FISHEYE` (8 params: fx, fy, cx, cy, k1-k4)
   - iPhone uses `OPENCV` (8 params: fx, fy, cx, cy, k1, k2, p1, p2)
   - Use undistorted images for simpler pinhole model

3. **Pose convention confusion**:
   - COLMAP: world-to-camera (invert for MapAnything)
   - NerfStudio: camera-to-world, OpenGL convention (convert Y/Z)
   - Our mapper handles all conversions automatically

4. **Out of memory with multi-view**:
   - Reduce `--num-views` 
   - Use smaller images (downscale with ScanNet++ toolkit)
   - Enable gradient checkpointing

---

## References

- [ScanNet++ Paper](https://arxiv.org/abs/2308.11417)
- [ScanNet++ Toolkit](https://github.com/scannetpp/scannetpp)
- [MapAnything Paper](https://arxiv.org/abs/2308.11417)
- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
