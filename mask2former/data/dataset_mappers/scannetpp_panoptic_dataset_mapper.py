# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for ScanNet++ dataset with multi-view support and camera pose handling
"""
ScanNet++ Panoptic Dataset Mapper for MapAnything + Mask2Former architecture.

This mapper:
1. Loads RGB images from ScanNet++ DSLR or iPhone captures
2. Loads camera intrinsics and extrinsics (poses) for MapAnything backbone
3. Loads 2D panoptic segmentation annotations
4. Implements view selection with minimal overlap between views of the same scene
5. Supports multi-view training by returning batches of related views

ScanNet++ Dataset Structure:
===========================
data/<scene_id>/
├── scans/
│   ├── mesh_aligned_0.05.ply          # 3D mesh
│   ├── mesh_aligned_0.05_semantic.ply # Semantic labels on mesh
│   ├── segments.json                   # Segment IDs per vertex
│   ├── segments_anno.json              # Instance annotations
│   └── scanner_poses.json              # Scanner positions (4x4 transforms)
├── dslr/
│   ├── resized_images/                 # Fisheye DSLR images (resized)
│   ├── resized_undistorted_images/     # Undistorted pinhole images
│   ├── resized_anon_masks/             # Anonymization masks
│   ├── colmap/
│   │   ├── cameras.txt                 # Camera intrinsics (OPENCV_FISHEYE)
│   │   ├── images.txt                  # Extrinsics (qvec, tvec per image)
│   │   └── points3D.txt                # 3D feature points
│   └── nerfstudio/
│       ├── transforms.json             # Poses in OpenGL/Blender convention
│       └── transforms_undistorted.json # For undistorted images
└── iphone/
    ├── rgb/                            # Extracted RGB frames
    ├── depth/                          # LiDAR depth maps (256x192)
    ├── pose_intrinsic_imu.json         # ARKit poses and intrinsics
    └── colmap/                         # COLMAP format (OPENCV model)

Camera Conventions:
==================
- COLMAP: world-to-camera transform, qvec (w,x,y,z) + tvec
- NerfStudio: camera-to-world transform, OpenGL convention
- MapAnything: camera-to-world transform needed for multi-view fusion
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

# Fix OpenCV multiprocessing issues
import cv2
cv2.setNumThreads(0)

__all__ = ["ScanNetPPPanopticDatasetMapper", "ScanNetPPMultiViewDatasetMapper"]

logger = logging.getLogger(__name__)


# ============================================================
# DEPTH RENDERING FROM 3D MESH
# ============================================================

def load_prerendered_depth(
    scene: 'ScanNetPPScene',
    frame: Dict,
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Load pre-rendered depth from ScanNet++ toolkit output.
    
    ScanNet++ provides a rendering script that outputs depth to:
        output_dir/SCENE_ID/dslr/render_depth/IMAGE_NAME.png
    
    The depth format is:
        - Single-channel uint16 PNG
        - Unit: millimeters (mm)
        - 0 means invalid depth
    
    Args:
        scene: ScanNetPPScene object
        frame: Frame dict with 'file_path'
        image_size: Optional (H, W) to resize depth map to match image
    
    Returns:
        Depth map [H, W] in METERS (converted from mm), or None if not found
    """
    # Get image filename
    image_path = Path(frame.get('file_path', ''))
    if not image_path.name:
        return None
    
    # Construct depth path: scene_dir/dslr/render_depth/IMAGE_NAME.png
    # The pre-rendered depth uses same name as image but .png extension
    image_stem = image_path.stem  # e.g., "DSC00001" from "DSC00001.JPG"
    
    # Try multiple possible locations for pre-rendered depth
    possible_depth_paths = [
        # Standard ScanNet++ toolkit output location
        scene.dslr_dir / 'render_depth' / f'{image_stem}.png',
        scene.dslr_dir / 'render_depth' / f'{image_stem}.JPG.png',
        # Alternative locations
        scene.scene_dir / 'dslr' / 'render_depth' / f'{image_stem}.png',
        scene.scene_dir / 'render_depth' / f'{image_stem}.png',
        # Undistorted depth location
        scene.dslr_dir / 'undistorted_render_depth' / f'{image_stem}.png',
    ]
    
    depth_path = None
    for path in possible_depth_paths:
        if path.exists():
            depth_path = path
            break
    
    if depth_path is None:
        # Depth not pre-rendered - this is OK, training works without it
        return None
    
    try:
        # Load uint16 PNG depth
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        if depth_mm is None:
            logger.warning(f"Failed to read depth: {depth_path}")
            return None
        
        # Convert from mm (uint16) to meters (float32)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        
        # 0 in uint16 means invalid - keep as 0.0 in float
        # (already 0.0 after division)
        
        # Resize if needed to match image size
        if image_size is not None:
            target_H, target_W = image_size
            if depth_m.shape[0] != target_H or depth_m.shape[1] != target_W:
                depth_m = cv2.resize(
                    depth_m, 
                    (target_W, target_H),  # cv2 uses (W, H)
                    interpolation=cv2.INTER_NEAREST  # Nearest for depth
                )
        
        return depth_m
        
    except Exception as e:
        logger.warning(f"Error loading depth from {depth_path}: {e}")
        return None


def load_or_render_depth(
    scene: 'ScanNetPPScene',
    frame: Dict,
    intrinsics: Dict,
    image_size: Tuple[int, int],
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """
    Load pre-rendered depth from ScanNet++ toolkit.
    
    This function loads depth from pre-rendered files created by:
        python -m common.render common/configs/render.yml
    
    The depth files are stored as uint16 PNG in mm units at:
        SCENE_ID/dslr/render_depth/IMAGE_NAME.png
    
    Args:
        scene: ScanNetPPScene object
        frame: Frame dict with 'file_path' and 'camera_to_world'
        intrinsics: Camera intrinsics dict (unused, kept for API compat)
        image_size: (H, W) for the depth map
        use_cache: Unused, kept for API compatibility
    
    Returns:
        Depth map [H, W] in meters, or None if not available
        
    Note:
        If depth is not available, training will still work - the spatial
        bridging in multi-view training will be disabled for that sample.
    """
    return load_prerendered_depth(scene, frame, image_size)


# ============================================================
# CAMERA POSE UTILITIES
# ============================================================

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to rotation matrix.
    
    COLMAP uses (w,x,y,z) quaternion convention.
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def colmap_to_camera_to_world(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP (world-to-camera) to camera-to-world 4x4 matrix.
    
    COLMAP stores world-to-camera transform:
    - qvec: quaternion (w,x,y,z)
    - tvec: translation vector
    
    Returns:
        4x4 camera-to-world transformation matrix
    """
    R_w2c = qvec2rotmat(qvec)
    t_w2c = tvec
    
    # world-to-camera: x_cam = R @ x_world + t
    # camera-to-world: x_world = R^T @ x_cam - R^T @ t
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    return T


def compute_camera_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """Compute Euclidean distance between camera centers.
    
    Args:
        pose1, pose2: 4x4 camera-to-world matrices
    
    Returns:
        Distance between camera centers in meters
    """
    center1 = pose1[:3, 3]
    center2 = pose2[:3, 3]
    return np.linalg.norm(center1 - center2)


def compute_viewing_direction_similarity(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """Compute cosine similarity between camera viewing directions.
    
    Viewing direction is the negative Z-axis of the camera frame.
    
    Returns:
        Cosine similarity in [-1, 1], where 1 = same direction
    """
    # Camera looks along -Z in OpenCV convention
    dir1 = -pose1[:3, 2]
    dir2 = -pose2[:3, 2]
    return np.dot(dir1, dir2)


def compute_view_overlap_score(pose1: np.ndarray, pose2: np.ndarray,
                               distance_threshold: float = 2.0) -> float:
    """Compute overlap score between two views.
    
    Higher score = more overlap (bad for diverse view selection).
    
    Args:
        pose1, pose2: 4x4 camera-to-world matrices
        distance_threshold: Distance (m) below which overlap is considered high
    
    Returns:
        Overlap score in [0, 1], where 0 = no overlap, 1 = maximum overlap
    """
    distance = compute_camera_distance(pose1, pose2)
    direction_similarity = compute_viewing_direction_similarity(pose1, pose2)
    
    # Normalize distance score (closer = higher score)
    distance_score = max(0, 1 - distance / distance_threshold)
    
    # Direction similarity already in [-1, 1], convert to [0, 1]
    direction_score = (direction_similarity + 1) / 2
    
    # Combined overlap score
    return distance_score * direction_score


# ============================================================
# COLMAP/NERFSTUDIO READERS
# ============================================================

def read_colmap_cameras(cameras_path: str) -> Dict:
    """Read COLMAP cameras.txt file.
    
    Returns:
        Dictionary with camera intrinsics and model type
    """
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,  # [fx, fy, cx, cy, k1, k2, k3, k4] for OPENCV_FISHEYE
            }
    return cameras


def read_colmap_images(images_path: str) -> Dict:
    """Read COLMAP images.txt file.
    
    Returns:
        Dictionary mapping image names to poses
    """
    images = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[j]) for j in range(1, 5)])  # w, x, y, z
        tvec = np.array([float(parts[j]) for j in range(5, 8)])
        camera_id = int(parts[8])
        image_name = parts[9]
        
        # Convert to camera-to-world
        c2w = colmap_to_camera_to_world(qvec, tvec)
        
        images[image_name] = {
            'image_id': image_id,
            'camera_id': camera_id,
            'qvec': qvec,
            'tvec': tvec,
            'camera_to_world': c2w,
        }
        
        # Skip the next line (2D points)
        i += 2
    
    return images


def read_nerfstudio_transforms(transforms_path: str) -> Tuple[Dict, List[Dict]]:
    """Read NerfStudio transforms.json file.
    
    Returns:
        Tuple of (camera_intrinsics, list of frame dicts with poses)
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    intrinsics = {
        'fl_x': data.get('fl_x'),
        'fl_y': data.get('fl_y'),
        'cx': data.get('cx'),
        'cy': data.get('cy'),
        'w': data.get('w'),
        'h': data.get('h'),
        'camera_model': data.get('camera_model', 'OPENCV_FISHEYE'),
        'k1': data.get('k1', 0),
        'k2': data.get('k2', 0),
        'k3': data.get('k3', 0),
        'k4': data.get('k4', 0),
    }
    
    frames = []
    for frame in data.get('frames', []):
        # NerfStudio uses OpenGL/Blender convention (camera-to-world)
        transform_matrix = np.array(frame['transform_matrix'])
        
        # Convert from OpenGL to OpenCV convention
        # OpenGL: +X right, +Y up, -Z forward
        # OpenCV: +X right, +Y down, +Z forward
        # Conversion: flip Y and Z
        opengl_to_opencv = np.diag([1, -1, -1, 1])
        c2w_opencv = transform_matrix @ opengl_to_opencv
        
        frames.append({
            'file_path': frame.get('file_path'),
            'camera_to_world': c2w_opencv,
            'camera_to_world_opengl': transform_matrix,
            'mask_path': frame.get('mask_path'),
            'is_bad': frame.get('is_bad', False),
        })
    
    return intrinsics, frames


# ============================================================
# VIEW SELECTION WITH MINIMAL OVERLAP
# ============================================================

def select_diverse_views(poses: List[np.ndarray], 
                         num_views: int,
                         min_distance: float = 0.3,
                         max_distance: float = 2.0,
                         seed: Optional[int] = None) -> List[int]:
    """Select views with HIGH visual overlap for query propagation.
    
    For multi-view training with query propagation, views need significant
    overlap (50-70%) so that warped attention masks are meaningful.
    
    This algorithm:
    1. Start with a random reference view
    2. Score candidates by: close distance + similar viewing direction
    3. Prefer nearby views looking at similar regions (high overlap)
    
    Args:
        poses: List of 4x4 camera-to-world matrices
        num_views: Number of views to select
        min_distance: Minimum distance (meters) between views (avoid identical)
        max_distance: Maximum distance (meters) — views beyond this have low overlap
        seed: Random seed for reproducibility
    
    Returns:
        List of indices of selected views
    """
    if len(poses) <= num_views:
        return list(range(len(poses)))
    
    rng = np.random.default_rng(seed)
    n = len(poses)
    
    # Start with a random reference view
    ref_idx = rng.integers(n)
    selected = [ref_idx]
    
    # Precompute distances and direction similarities from reference
    scores = np.full(n, -np.inf)
    for i in range(n):
        if i == ref_idx:
            continue
        dist = compute_camera_distance(poses[i], poses[ref_idx])
        dir_sim = compute_viewing_direction_similarity(poses[i], poses[ref_idx])
        
        # Skip views that are too close (near-duplicate) or too far (no overlap)
        if dist < min_distance:
            continue
        if dist > max_distance * 3:  # Hard cutoff at 3x max_distance
            continue
        
        # Overlap score: prefer close distance + similar viewing direction
        # Distance score: peaks at ideal_dist, falls off on both sides
        ideal_dist = (min_distance + max_distance) / 2  # Sweet spot
        dist_score = np.exp(-((dist - ideal_dist) / max_distance) ** 2)
        
        # Direction score: prefer views looking in similar direction (overlap)
        # dir_sim in [-1, 1], we want close to 1 (same direction)
        dir_score = max(0, (dir_sim + 1) / 2)  # Map to [0, 1]
        
        # Combined score: 60% direction (overlap), 40% distance
        scores[i] = 0.6 * dir_score + 0.4 * dist_score
    
    while len(selected) < num_views:
        # Among unselected, find best score
        best_idx = -1
        best_score = -np.inf
        for i in range(n):
            if i in selected:
                continue
            if scores[i] <= -np.inf:
                continue
            
            # Also check minimum distance to ALL already-selected views
            too_close = False
            for j in selected:
                if compute_camera_distance(poses[i], poses[j]) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
            
            # Add small random jitter for diversity across epochs
            jittered_score = scores[i] + rng.uniform(0, 0.1)
            if jittered_score > best_score:
                best_score = jittered_score
                best_idx = i
        
        if best_idx < 0:
            # No valid candidates — relax constraints and pick randomly
            remaining = [i for i in range(n) if i not in selected]
            if remaining:
                selected.append(rng.choice(remaining))
            else:
                break
        else:
            selected.append(best_idx)
    
    return selected


def select_views_for_scene(frames: List[Dict],
                           num_views: int = 8,
                           min_distance: float = 0.3,
                           max_distance: float = 2.0,
                           seed: Optional[int] = None) -> List[int]:
    """Select views for a scene with HIGH visual overlap for query propagation.
    
    Uses overlap-aware selection that prefers nearby views with similar
    viewing directions, ensuring warped attention masks are meaningful.
    
    Args:
        frames: List of frame dicts with 'camera_to_world' and optional 'is_bad'
        num_views: Target number of views
        min_distance: Minimum camera distance in meters (avoid duplicates)
        max_distance: Maximum camera distance in meters (beyond = low overlap)
        seed: Random seed
    
    Returns:
        List of selected frame indices
    """
    # Filter out bad frames
    valid_indices = [i for i, f in enumerate(frames) if not f.get('is_bad', False)]
    
    if len(valid_indices) == 0:
        logger.warning("No valid frames found in scene!")
        return []
    
    # Get poses for valid frames
    valid_poses = [frames[i]['camera_to_world'] for i in valid_indices]
    
    # Use overlap-aware view selection
    selected_local = select_diverse_views(
        valid_poses, num_views, min_distance, max_distance, seed
    )
    
    # Map back to original indices
    selected = [valid_indices[i] for i in selected_local]
    
    return selected


# ============================================================
# INTRINSIC MATRIX CONSTRUCTION
# ============================================================

def build_intrinsic_matrix(intrinsics: Dict) -> np.ndarray:
    """Build 3x3 intrinsic matrix from camera parameters.
    
    Args:
        intrinsics: Dict with fl_x, fl_y, cx, cy
    
    Returns:
        3x3 intrinsic matrix K
    """
    K = np.array([
        [intrinsics['fl_x'], 0, intrinsics['cx']],
        [0, intrinsics['fl_y'], intrinsics['cy']],
        [0, 0, 1]
    ])
    return K


def build_intrinsic_matrix_4x4(intrinsics: Dict) -> np.ndarray:
    """Build 4x4 intrinsic matrix for projection.
    
    Used for depth unprojection in MapAnything.
    """
    K = np.eye(4)
    K[0, 0] = intrinsics['fl_x']
    K[1, 1] = intrinsics['fl_y']
    K[0, 2] = intrinsics['cx']
    K[1, 2] = intrinsics['cy']
    return K


# ============================================================
# SCANNET++ SCENE CLASS
# ============================================================

class ScanNetPPScene:
    """Helper class to access ScanNet++ scene data.
    
    Mirrors the structure from scannetpp/common/scene_release.py
    """
    
    def __init__(self, scene_id: str, data_root: str):
        self.scene_id = scene_id
        self.data_root = Path(data_root)
        # ScanNet++ structure: <data_root>/data/<scene_id>/
        self.scene_dir = self.data_root / 'data' / scene_id
    
    # DSLR paths
    @property
    def dslr_dir(self) -> Path:
        return self.scene_dir / 'dslr'
    
    @property
    def dslr_resized_dir(self) -> Path:
        return self.dslr_dir / 'resized_images'
    
    @property
    def dslr_undistorted_dir(self) -> Path:
        return self.dslr_dir / 'resized_undistorted_images'
    
    @property
    def dslr_mask_dir(self) -> Path:
        return self.dslr_dir / 'resized_anon_masks'
    
    @property
    def dslr_colmap_dir(self) -> Path:
        return self.dslr_dir / 'colmap'
    
    @property
    def dslr_nerfstudio_transforms(self) -> Path:
        return self.dslr_dir / 'nerfstudio' / 'transforms.json'
    
    @property
    def dslr_nerfstudio_transforms_undistorted(self) -> Path:
        return self.dslr_dir / 'nerfstudio' / 'transforms_undistorted.json'
    
    # iPhone paths
    @property
    def iphone_dir(self) -> Path:
        return self.scene_dir / 'iphone'
    
    @property
    def iphone_rgb_dir(self) -> Path:
        return self.iphone_dir / 'rgb'
    
    @property
    def iphone_depth_dir(self) -> Path:
        return self.iphone_dir / 'depth'
    
    @property
    def iphone_colmap_dir(self) -> Path:
        return self.iphone_dir / 'colmap'
    
    @property
    def iphone_pose_intrinsic_path(self) -> Path:
        return self.iphone_dir / 'pose_intrinsic_imu.json'
    
    @property
    def iphone_nerfstudio_transforms(self) -> Path:
        return self.iphone_dir / 'nerfstudio' / 'transforms.json'
    
    # Scan/mesh paths
    @property
    def scans_dir(self) -> Path:
        return self.scene_dir / 'scans'
    
    @property
    def mesh_path(self) -> Path:
        return self.scans_dir / 'mesh_aligned_0.05.ply'
    
    @property
    def semantic_mesh_path(self) -> Path:
        return self.scans_dir / 'mesh_aligned_0.05_semantic.ply'
    
    @property
    def segments_path(self) -> Path:
        return self.scans_dir / 'segments.json'
    
    @property
    def segments_anno_path(self) -> Path:
        return self.scans_dir / 'segments_anno.json'


# ============================================================
# SINGLE-VIEW DATASET MAPPER
# ============================================================

class ScanNetPPPanopticDatasetMapper:
    """
    Dataset mapper for ScanNet++ panoptic segmentation (single-view).
    
    This mapper:
    1. Loads RGB image from ScanNet++ scene
    2. Loads camera intrinsics and extrinsics
    3. Loads 2D panoptic segmentation from rasterized 3D annotations
    4. Applies augmentations consistently to image and labels
    5. Returns data in format expected by MapAnything + Mask2Former
    
    Usage:
        mapper = ScanNetPPPanopticDatasetMapper.from_config(cfg, is_train=True)
        data = mapper(dataset_dict)
    
    Expected input dataset_dict:
        {
            'file_name': str,        # Path to RGB image
            'scene_id': str,         # ScanNet++ scene ID
            'image_id': int,         # Unique image identifier
            'height': int,           # Image height
            'width': int,            # Image width
            'pan_seg_file_name': str,  # Path to panoptic segmentation PNG
            'segments_info': List[dict],  # Segment metadata
            # Optional pose information
            'camera_to_world': Optional[np.ndarray],  # 4x4 pose matrix
            'intrinsics': Optional[dict],             # Camera intrinsics
        }
    """
    
    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List,
        image_format: str,
        ignore_label: int,
        size_divisibility: int,
        use_undistorted: bool = True,
        image_type: str = 'dslr',
    ):
        """
        Args:
            is_train: Whether in training mode
            augmentations: List of augmentation transforms
            image_format: Image format (RGB or BGR)
            ignore_label: Label to ignore during training
            size_divisibility: Pad images to be divisible by this value
            use_undistorted: Whether to use undistorted images (recommended)
            image_type: 'dslr' or 'iphone'
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.use_undistorted = use_undistorted
        self.image_type = image_type
        
        logger.info(
            f"[ScanNetPPPanopticDatasetMapper] "
            f"is_train={is_train}, image_type={image_type}, "
            f"use_undistorted={use_undistorted}"
        )
    
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # Build augmentations
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice"
                ),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
        else:
            augs = [
                T.ResizeShortestEdge(
                    short_edge_length=800,
                    max_size=1333,
                    sample_style="choice"
                ),
            ]
        
        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if hasattr(cfg.INPUT, 'SIZE_DIVISIBILITY') else 0,
            "use_undistorted": cfg.INPUT.get('USE_UNDISTORTED', True),
            "image_type": cfg.INPUT.get('IMAGE_TYPE', 'dslr'),
        }
    
    def __call__(self, dataset_dict: Dict) -> Dict:
        """
        Process a single sample.
        
        Args:
            dataset_dict: Input sample with image path and annotations
        
        Returns:
            Processed sample ready for training
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        # Load panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict["pan_seg_file_name"], "RGB")
            segments_info = dataset_dict.get("segments_info", [])
        else:
            pan_seg_gt = None
            segments_info = []
        
        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = T.apply_transform_gens(self.tfm_gens, aug_input)[1]
        image = aug_input.image
        
        if pan_seg_gt is not None:
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
        
        # Convert panoptic RGB to ID
        # CRITICAL FIX: The panoptic PNGs were saved with cv2.imwrite(path, rgb[:,:,::-1])
        # which swaps R↔B channels. When read back as RGB by PIL/read_image, channels are
        # swapped: what should be R (least significant byte) is in B channel.
        # So we must use: B + G*256 + R*65536 (NOT the standard R + G*256 + B*65536)
        if pan_seg_gt is not None:
            pan_seg_gt = (
                pan_seg_gt[:, :, 2].astype(np.int32) +          # B channel = least significant byte
                pan_seg_gt[:, :, 1].astype(np.int32) * 256 +    # G channel = middle byte
                pan_seg_gt[:, :, 0].astype(np.int32) * 65536    # R channel = most significant byte
            )
        
        # Convert to tensors
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if pan_seg_gt is not None:
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
        
        # Padding
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1] % self.size_divisibility,
                0,
                self.size_divisibility - image_size[0] % self.size_divisibility,
            ]
            if padding_size[1] == self.size_divisibility:
                padding_size[1] = 0
            if padding_size[3] == self.size_divisibility:
                padding_size[3] = 0
            
            if any(p > 0 for p in padding_size):
                import torch.nn.functional as F
                image = F.pad(image, padding_size, value=128).contiguous()
                if pan_seg_gt is not None:
                    pan_seg_gt = F.pad(pan_seg_gt, padding_size, value=0).contiguous()
        
        image_shape = (image.shape[-2], image.shape[-1])
        
        # Store in dataset dict
        dataset_dict["image"] = image
        
        # Process camera pose if available
        if "camera_to_world" in dataset_dict:
            c2w = dataset_dict["camera_to_world"]
            if isinstance(c2w, np.ndarray):
                c2w = torch.from_numpy(c2w).float()
            dataset_dict["camera_to_world"] = c2w
        
        if "intrinsics" in dataset_dict:
            intrinsics = dataset_dict["intrinsics"]
            if isinstance(intrinsics, dict):
                K = build_intrinsic_matrix(intrinsics)
                dataset_dict["intrinsic_matrix"] = torch.from_numpy(K).float()
        
        # Process panoptic annotations into instances
        if pan_seg_gt is not None and self.is_train:
            pan_seg_np = pan_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = []
            masks = []
            
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info.get("iscrowd", False):
                    classes.append(class_id)
                    masks.append(pan_seg_np == segment_info["id"])
            
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            
            if len(masks) == 0:
                instances.gt_masks = torch.zeros((0, pan_seg_np.shape[0], pan_seg_np.shape[1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
            
            dataset_dict["instances"] = instances
        
        return dataset_dict


# ============================================================
# MULTI-VIEW DATASET MAPPER
# ============================================================

class ScanNetPPMultiViewDatasetMapper:
    """
    Multi-view dataset mapper for MapAnything with ScanNet++.
    
    This mapper returns multiple views of the same scene with:
    1. Diverse camera viewpoints (minimal overlap)
    2. Consistent camera poses for multi-view fusion
    3. Panoptic annotations for each view
    
    Key features:
    - Implements farthest point sampling for view selection
    - Maintains camera-to-world transforms for MapAnything backbone
    - Supports variable number of views per training sample
    
    The output format matches MapAnything's MultiViewTransformerInput:
        views = [{
            'img': [B, 3, H, W],
            'camera_to_world': [B, 4, 4],
            'intrinsics': [B, 4, 4],
            ...
        }]
    """
    
    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List,
        image_format: str,
        ignore_label: int,
        size_divisibility: int,
        num_views: int = 4,
        min_view_distance: float = 0.3,
        max_view_distance: float = 2.0,
        use_undistorted: bool = True,
        image_type: str = 'dslr',
        data_root: str = "",
        panoptic_root: str = "",
        load_depth: bool = True,
        depth_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            num_views: Number of views to sample per scene
            min_view_distance: Minimum distance (meters) between views
            panoptic_root: Root directory for 2D panoptic annotations
            load_depth: Whether to load/render depth from mesh
            depth_cache_dir: Optional global cache dir for depth maps
            Other args same as ScanNetPPPanopticDatasetMapper
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_views = num_views
        self.min_view_distance = min_view_distance
        self.max_view_distance = max_view_distance
        self.use_undistorted = use_undistorted
        self.image_type = image_type
        self.data_root = data_root
        self.panoptic_root = panoptic_root
        self.load_depth = load_depth
        self.depth_cache_dir = depth_cache_dir
        
        # Cache for scene data
        self._scene_cache = {}
        
        logger.info(
            f"[ScanNetPPMultiViewDatasetMapper] "
            f"num_views={num_views}, min_distance={min_view_distance}m, "
            f"max_distance={max_view_distance}m, "
            f"load_depth={load_depth}, data_root={data_root}"
        )
    
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # Build augmentations (same for all views in a batch)
        # Use config values for augmentation sizes (not hardcoded)
        train_sizes = cfg.INPUT.MIN_SIZE_TRAIN if hasattr(cfg.INPUT, 'MIN_SIZE_TRAIN') else (640, 672, 704, 736, 768, 800)
        train_max = cfg.INPUT.MAX_SIZE_TRAIN if hasattr(cfg.INPUT, 'MAX_SIZE_TRAIN') else 1333
        test_size = cfg.INPUT.MIN_SIZE_TEST if hasattr(cfg.INPUT, 'MIN_SIZE_TEST') else 800
        test_max = cfg.INPUT.MAX_SIZE_TEST if hasattr(cfg.INPUT, 'MAX_SIZE_TEST') else 1333
        
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    short_edge_length=train_sizes,
                    max_size=train_max,
                    sample_style="choice"
                ),
            ]
            # Note: Random flip disabled for multi-view to maintain pose consistency
        else:
            augs = [
                T.ResizeShortestEdge(
                    short_edge_length=test_size,
                    max_size=test_max,
                    sample_style="choice"
                ),
            ]
        
        # Get data paths from config OR from metadata (set by register_datasets.py)
        # First try cfg, then fall back to metadata
        data_root = ""
        panoptic_root = ""
        
        if hasattr(cfg.DATASETS, 'SCANNETPP_ROOT') and cfg.DATASETS.SCANNETPP_ROOT:
            data_root = cfg.DATASETS.SCANNETPP_ROOT
        if hasattr(cfg.DATASETS, 'SCANNETPP_PANOPTIC_ROOT') and cfg.DATASETS.SCANNETPP_PANOPTIC_ROOT:
            panoptic_root = cfg.DATASETS.SCANNETPP_PANOPTIC_ROOT
        
        # If not set in config, try to get from MetadataCatalog (set by register_datasets.py)
        if not data_root or not panoptic_root:
            try:
                from detectron2.data import MetadataCatalog
                dataset_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]
                meta = MetadataCatalog.get(dataset_name)
                if not data_root and hasattr(meta, 'image_root'):
                    data_root = meta.image_root
                if not panoptic_root and hasattr(meta, 'panoptic_root'):
                    panoptic_root = meta.panoptic_root
            except Exception as e:
                logger.warning(f"Could not get paths from MetadataCatalog: {e}")
        
        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "size_divisibility": cfg.INPUT.get('SIZE_DIVISIBILITY', 0),
            "num_views": cfg.MODEL.MULTIVIEW.NUM_VIEWS if hasattr(cfg.MODEL, 'MULTIVIEW') else cfg.INPUT.get('NUM_VIEWS', 4),
            "min_view_distance": cfg.MODEL.MULTIVIEW.MIN_CAMERA_DISTANCE if hasattr(cfg.MODEL, 'MULTIVIEW') else cfg.INPUT.get('MIN_VIEW_DISTANCE', 0.3),
            "max_view_distance": cfg.MODEL.MULTIVIEW.MAX_CAMERA_DISTANCE if hasattr(cfg.MODEL, 'MULTIVIEW') else cfg.INPUT.get('MAX_VIEW_DISTANCE', 2.0),
            "use_undistorted": cfg.INPUT.get('USE_UNDISTORTED', True),
            "image_type": cfg.INPUT.get('IMAGE_TYPE', 'dslr'),
            "data_root": data_root,
            "panoptic_root": panoptic_root,
            # Disable depth loading by default to avoid Open3D segfaults in workers
            "load_depth": cfg.INPUT.get('LOAD_DEPTH', False),
            "depth_cache_dir": cfg.INPUT.get('DEPTH_CACHE_DIR', None),
        }
    
    def _load_scene_data(self, scene_id: str) -> Dict:
        """Load and cache scene camera data."""
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]
        
        scene = ScanNetPPScene(scene_id, self.data_root)
        
        # Load camera data
        if self.image_type == 'dslr':
            transforms_path = (
                scene.dslr_nerfstudio_transforms_undistorted
                if self.use_undistorted
                else scene.dslr_nerfstudio_transforms
            )
            image_dir = (
                scene.dslr_undistorted_dir
                if self.use_undistorted
                else scene.dslr_resized_dir
            )
        else:
            transforms_path = scene.iphone_nerfstudio_transforms
            image_dir = scene.iphone_rgb_dir
        
        intrinsics = None
        frames = []
        
        # Try NerfStudio transforms first
        if transforms_path.exists():
            intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
        else:
            # Try COLMAP as fallback
            colmap_dir = (
                scene.dslr_colmap_dir
                if self.image_type == 'dslr'
                else scene.iphone_colmap_dir
            )
            cameras_file = colmap_dir / 'cameras.txt'
            images_file = colmap_dir / 'images.txt'
            
            if cameras_file.exists() and images_file.exists():
                cameras = read_colmap_cameras(str(cameras_file))
                images = read_colmap_images(str(images_file))
                
                # Convert to frame format
                cam = list(cameras.values())[0]
                intrinsics = {
                    'fl_x': cam['params'][0],
                    'fl_y': cam['params'][1],
                    'cx': cam['params'][2],
                    'cy': cam['params'][3],
                    'w': cam['width'],
                    'h': cam['height'],
                    'camera_model': cam['model'],
                }
                if len(cam['params']) > 4:
                    intrinsics['k1'] = cam['params'][4]
                    intrinsics['k2'] = cam['params'][5]
                    if len(cam['params']) > 6:
                        intrinsics['k3'] = cam['params'][6]
                        intrinsics['k4'] = cam['params'][7]
                
                frames = []
                for name, data in images.items():
                    frames.append({
                        'file_path': str(image_dir / name),
                        'camera_to_world': data['camera_to_world'],
                        'is_bad': False,
                    })
            else:
                logger.warning(
                    f"Scene {scene_id}: No NerfStudio transforms at {transforms_path.absolute()} "
                    f"and no COLMAP files at {colmap_dir.absolute()}"
                )
        
        if intrinsics is None or len(frames) == 0:
            logger.warning(f"No valid frames found in scene {scene_id}!")
        
        scene_data = {
            'scene': scene,
            'intrinsics': intrinsics,
            'frames': frames,
            'image_dir': image_dir,
        }
        
        self._scene_cache[scene_id] = scene_data
        return scene_data
    
    def __call__(self, dataset_dict: Dict) -> Dict:
        """
        Process a multi-view sample from a scene.
        
        Input dataset_dict should have at least 'scene_id'.
        
        Returns:
            Dict with 'views' list, each containing:
            - image: [3, H, W] tensor
            - camera_pose: [4, 4] tensor
            - camera_intrinsic: [3, 3] tensor
            - sem_seg: [H, W] tensor (if available)
            - instances: Instances object (if available)
            - file_name: str
            
            Returns None if scene has no valid camera data (will be filtered out)
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        scene_id = dataset_dict['scene_id']
        
        # Load scene data
        scene_data = self._load_scene_data(scene_id)
        intrinsics = scene_data['intrinsics']
        frames = scene_data['frames']
        image_dir = scene_data['image_dir']
        scene = scene_data['scene']
        
        # Check if we have valid camera data
        if intrinsics is None or len(frames) == 0:
            logger.warning(f"Skipping scene {scene_id}: No valid camera data or frames")
            return None
        
        # Select diverse views
        if self.is_train:
            seed = np.random.randint(0, 10000)
        else:
            seed = hash(scene_id) % 10000  # Deterministic for eval
        
        selected_indices = select_views_for_scene(
            frames,
            num_views=self.num_views,
            min_distance=self.min_view_distance,
            max_distance=getattr(self, 'max_view_distance', 2.0),
            seed=seed,
        )
        
        # Get panoptic annotation directory if it exists
        panoptic_dir = None
        if hasattr(self, 'panoptic_root') and self.panoptic_root:
            panoptic_dir = Path(self.panoptic_root) / scene_id
        
        # Build 3x3 intrinsic matrix
        K = build_intrinsic_matrix(intrinsics)
        K_tensor = torch.from_numpy(K).float()
        
        # For multi-view consistency, sample target size ONCE for all views
        # This ensures all views are resized to the same dimensions
        if self.is_train and len(self.tfm_gens) > 0:
            # Sample a single target size for all views in this scene
            # For ResizeShortestEdge with choice, manually sample from the list
            import random
            rng = random.Random(seed)  # Use same seed as view selection for reproducibility
            
            # Extract the size choices from ResizeShortestEdge (first transform)
            if hasattr(self.tfm_gens[0], 'short_edge_length'):
                if isinstance(self.tfm_gens[0].short_edge_length, (list, tuple)):
                    target_size = rng.choice(self.tfm_gens[0].short_edge_length)
                else:
                    target_size = self.tfm_gens[0].short_edge_length
                max_size = self.tfm_gens[0].max_size
            else:
                target_size = 800  # fallback
                max_size = 1333
        else:
            # Eval mode: use deterministic size
            target_size = 800
            max_size = 1333
        
        # First pass: resize all images and find the maximum dimensions
        # This ensures all views will have the exact same final size
        resized_images = []
        for idx in selected_indices:
            frame = frames[idx]
            image_path = frame.get('file_path')
            if image_path and not Path(image_path).is_absolute():
                image_path = str(image_dir / Path(image_path).name)
            
            image = utils.read_image(image_path, format=self.img_format)
            
            # Apply resize transform
            resize_tfm = T.ResizeShortestEdge(target_size, max_size)
            aug_input = T.AugInput(image)
            transforms = resize_tfm.get_transform(aug_input.image)
            image = transforms.apply_image(image)
            
            resized_images.append({
                'image': image,
                'frame': frame,
                'image_path': image_path,
                'transforms': transforms,
            })
        
        # Find maximum height and width across all resized views
        max_h = max(img_data['image'].shape[0] for img_data in resized_images)
        max_w = max(img_data['image'].shape[1] for img_data in resized_images)
        
        # Apply size divisibility constraint
        if self.size_divisibility > 0:
            max_h = int(np.ceil(max_h / self.size_divisibility) * self.size_divisibility)
            max_w = int(np.ceil(max_w / self.size_divisibility) * self.size_divisibility)
        
        # Process each selected view
        views = []
        
        for img_data in resized_images:
            image = img_data['image']
            frame = img_data['frame']
            image_path = img_data['image_path']
            transforms = img_data['transforms']
            
            # Pad to common size (bottom and right padding)
            pad_h = max_h - image.shape[0]
            pad_w = max_w - image.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                image = np.pad(
                    image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode='constant',
                    constant_values=128
                )
            
            image_shape = image.shape[:2]  # H, W (should be max_h x max_w after padding)
            
            # Convert to tensor
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            
            # Get pose
            c2w = frame['camera_to_world']
            if isinstance(c2w, np.ndarray):
                c2w = torch.from_numpy(c2w).float()
            
            view_data = {
                'image': image,
                'camera_pose': c2w,
                'camera_intrinsic': K_tensor.clone(),
                'file_name': str(image_path),
            }
            
            # Load/render depth from 3D mesh
            if self.load_depth:
                # Get original image size before padding for depth rendering
                orig_H, orig_W = image_shape
                
                # Render depth from mesh
                depth = load_or_render_depth(
                    scene=scene,
                    frame=frame,
                    intrinsics=intrinsics,
                    image_size=(orig_H, orig_W),
                    use_cache=True,
                )
                
                if depth is not None:
                    # Resize depth to match the padded image dimensions
                    if depth.shape != (max_h, max_w):
                        # First resize to the resized image size (before padding)
                        resized_h = image_shape[0] - pad_h
                        resized_w = image_shape[1] - pad_w
                        
                        if (depth.shape[0], depth.shape[1]) != (resized_h, resized_w):
                            depth = cv2.resize(
                                depth,
                                (resized_w, resized_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # Then apply same padding
                        if pad_h > 0 or pad_w > 0:
                            depth = np.pad(
                                depth,
                                ((0, pad_h), (0, pad_w)),
                                mode='constant',
                                constant_values=0
                            )
                    
                    # Convert to tensor [1, H, W]
                    depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
                    view_data['depth'] = depth_tensor
            
            # Load panoptic annotations if available
            if panoptic_dir is not None:
                stem = Path(image_path).stem
                pan_seg_path = panoptic_dir / f"{stem}.png"
                
                if pan_seg_path.exists():
                    pan_seg_gt = utils.read_image(str(pan_seg_path), "RGB")
                    
                    # Apply transforms if available
                    if transforms is not None:
                        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
                    
                    # Convert RGB to ID
                    # CRITICAL FIX: PNGs were saved with cv2.imwrite(path, rgb[:,:,::-1])
                    # which swaps R↔B. When read back as RGB, B has least significant byte.
                    # Use: B + G*256 + R*65536 (NOT standard R + G*256 + B*65536)
                    pan_seg_gt = (
                        pan_seg_gt[:, :, 2].astype(np.int32) +          # B = LSB
                        pan_seg_gt[:, :, 1].astype(np.int32) * 256 +    # G = middle
                        pan_seg_gt[:, :, 0].astype(np.int32) * 65536    # R = MSB
                    )

                    # Apply same padding as image
                    if pad_h > 0 or pad_w > 0:
                        pan_seg_gt = np.pad(
                            pan_seg_gt,
                            ((0, pad_h), (0, pad_w)),
                            mode='constant',
                            constant_values=self.ignore_label
                        )
                    
                    view_data['sem_seg'] = torch.as_tensor(pan_seg_gt.astype("long"))
                    
                    # Load segments_info if available
                    info_path = panoptic_dir / f"{stem}.json"
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            segments_info = json.load(f)
                        
                        # Create Instances object
                        if self.is_train:
                            instances = Instances((image.shape[-2], image.shape[-1]))
                            classes = []
                            masks = []
                            
                            for seg_info in segments_info:
                                class_id = seg_info["category_id"]
                                if not seg_info.get("iscrowd", False):
                                    classes.append(class_id)
                                    masks.append(pan_seg_gt == seg_info["id"])
                            
                            # Cap GT instances to leave room for "no object" queries.
                            # With 100 queries, 95 GT objects leaves only 5 unmatched,
                            # creating extreme CE gradients and NaN/Inf in AMP fp16.
                            # Keep the largest segments (by area) for best supervision.
                            MAX_GT_INSTANCES = 80  # Leave ≥20 queries for no-object
                            if len(classes) > MAX_GT_INSTANCES:
                                # Sort by mask area (descending), keep largest
                                areas = [m.sum() for m in masks]
                                sorted_indices = sorted(range(len(areas)), 
                                                       key=lambda i: areas[i], reverse=True)
                                sorted_indices = sorted_indices[:MAX_GT_INSTANCES]
                                sorted_indices.sort()  # Preserve original order
                                classes = [classes[i] for i in sorted_indices]
                                masks = [masks[i] for i in sorted_indices]
                            
                            if len(classes) > 0:
                                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
                                masks = BitMasks(
                                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                                )
                                instances.gt_masks = masks.tensor
                                instances.gt_boxes = masks.get_bounding_boxes()
                            else:
                                instances.gt_classes = torch.zeros(0, dtype=torch.int64)
                                instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
                                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                            
                            view_data['instances'] = instances
            
            views.append(view_data)
        
        # Verify all views have the same image dimensions
        if len(views) > 0:
            first_shape = views[0]['image'].shape
            for i, view in enumerate(views):
                if view['image'].shape != first_shape:
                    logger.error(
                        f"Scene {scene_id}: Size mismatch in view {i}: "
                        f"expected {first_shape}, got {view['image'].shape}. "
                        f"max_h={max_h}, max_w={max_w}"
                    )
                    # This should not happen with the two-pass approach
                    raise RuntimeError(f"Multi-view size mismatch: {first_shape} vs {view['image'].shape}")
        
        # Return in format expected by multi_view_collate_fn
        return {
            'scene_id': scene_id,
            'views': views,
            'selected_view_indices': selected_indices,
        }


# ============================================================
# DATASET REGISTRATION UTILITIES
# ============================================================

def load_scannetpp_panoptic_json(json_path: str, data_root: str, 
                                  image_type: str = 'dslr',
                                  use_undistorted: bool = True) -> List[Dict]:
    """
    Load ScanNet++ dataset annotations from JSON.
    
    Expected JSON format:
    {
        "scenes": [
            {
                "scene_id": "abc123",
                "images": [
                    {
                        "file_name": "DSC00001.JPG",
                        "pan_seg_file_name": "DSC00001.png",
                        "segments_info": [...]
                    },
                    ...
                ]
            },
            ...
        ]
    }
    
    Returns:
        List of dataset dicts in Detectron2 format
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset_dicts = []
    image_id = 0
    
    for scene_data in data.get('scenes', []):
        scene_id = scene_data['scene_id']
        scene = ScanNetPPScene(scene_id, data_root)
        
        # Determine image directory
        if image_type == 'dslr':
            image_dir = (
                scene.dslr_undistorted_dir if use_undistorted
                else scene.dslr_resized_dir
            )
        else:
            image_dir = scene.iphone_rgb_dir
        
        for img_data in scene_data.get('images', []):
            record = {
                'image_id': image_id,
                'scene_id': scene_id,
                'file_name': str(image_dir / img_data['file_name']),
                'height': img_data.get('height'),
                'width': img_data.get('width'),
            }
            
            if 'pan_seg_file_name' in img_data:
                record['pan_seg_file_name'] = img_data['pan_seg_file_name']
                record['segments_info'] = img_data.get('segments_info', [])
            
            dataset_dicts.append(record)
            image_id += 1
    
    return dataset_dicts


def register_scannetpp_panoptic(
    name: str,
    data_root: str,
    split_file: str,
    panoptic_dir: str,
    image_type: str = 'dslr',
    use_undistorted: bool = True,
    num_classes: int = 100,
):
    """
    Register a ScanNet++ panoptic dataset with Detectron2.
    
    Args:
        name: Dataset name for registration
        data_root: Root directory of ScanNet++ data
        split_file: Path to split file (e.g., nvs_sem_train.txt)
        panoptic_dir: Directory with rasterized 2D panoptic annotations
        image_type: 'dslr' or 'iphone'
        use_undistorted: Use undistorted images
        num_classes: Number of semantic classes
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    def load_dataset():
        return _load_scannetpp_panoptic_dataset(
            data_root, split_file, panoptic_dir, image_type, use_undistorted
        )
    
    DatasetCatalog.register(name, load_dataset)
    
    MetadataCatalog.get(name).set(
        stuff_classes=["class_" + str(i) for i in range(num_classes)],
        thing_classes=["thing_" + str(i) for i in range(num_classes)],
        ignore_label=255,
        image_type=image_type,
        data_root=data_root,
        evaluator_type="coco_panoptic_seg",
        panoptic_root=panoptic_dir,
        image_root=data_root,
        label_divisor=10000, 
        stuff_dataset_id_to_contiguous_id={},  # Required by COCOPanopticEvaluator
        thing_dataset_id_to_contiguous_id={},  # Required by COCOPanopticEvaluator
    )


def _load_scannetpp_panoptic_dataset(
    data_root: str,
    split_file: str,
    panoptic_dir: str,
    image_type: str,
    use_undistorted: bool,
) -> List[Dict]:
    """Internal function to load dataset."""
    # Read split file
    with open(split_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    
    dataset_dicts = []
    image_id = 0
    
    for scene_id in scene_ids:
        scene = ScanNetPPScene(scene_id, data_root)
        
        # Get image directory and list
        if image_type == 'dslr':
            image_dir = (
                scene.dslr_undistorted_dir if use_undistorted
                else scene.dslr_resized_dir
            )
            transforms_path = (
                scene.dslr_nerfstudio_transforms_undistorted if use_undistorted
                else scene.dslr_nerfstudio_transforms
            )
        else:
            image_dir = scene.iphone_rgb_dir
            transforms_path = scene.iphone_nerfstudio_transforms
        
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            continue
        
        # Get camera data
        intrinsics, frames = None, []
        if transforms_path.exists():
            intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
        
        # Get panoptic annotations
        scene_panoptic_dir = Path(panoptic_dir) / scene_id
        
        # List all images
        image_files = sorted(image_dir.glob('*.JPG')) + sorted(image_dir.glob('*.jpg'))
        
        frame_lookup = {Path(f['file_path']).stem: f for f in frames}
        
        for img_path in image_files:
            record = {
                'image_id': image_id,
                'scene_id': scene_id,
                'file_name': str(img_path),
            }
            
            # Get image dimensions from camera intrinsics (fast) instead of reading file
            if intrinsics and 'h' in intrinsics and 'w' in intrinsics:
                record['height'] = int(intrinsics['h'])
                record['width'] = int(intrinsics['w'])
            else:
                # Fallback: use default DSLR dimensions (will be verified during actual loading)
                record['height'] = 6048
                record['width'] = 4024
            
            # Add camera pose if available
            stem = img_path.stem
            if stem in frame_lookup:
                frame = frame_lookup[stem]
                record['camera_to_world'] = frame['camera_to_world']
            
            if intrinsics:
                record['intrinsics'] = intrinsics
            
            # Add panoptic annotation if available
            pan_seg_path = scene_panoptic_dir / f"{stem}.png"
            if pan_seg_path.exists():
                record['pan_seg_file_name'] = str(pan_seg_path)
                
                # Load segments_info if available
                info_path = scene_panoptic_dir / f"{stem}.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        record['segments_info'] = json.load(f)
                else:
                    record['segments_info'] = []
            
            dataset_dicts.append(record)
            image_id += 1
    
    logger.info(f"Loaded {len(dataset_dicts)} images from {len(scene_ids)} scenes")
    return dataset_dicts
