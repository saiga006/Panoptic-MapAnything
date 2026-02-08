#!/usr/bin/env python3
"""
ScanNet++ Panoptic Label Generator

This script generates 2D panoptic segmentation labels from ScanNet++ 3D mesh annotations.

ScanNet++ provides:
- mesh_aligned_0.05.ply: High-quality 3D mesh
- mesh_aligned_0.05_semantic.ply: Mesh with per-vertex semantic labels (in vertex colors)
- segments.json: Segment IDs per vertex
- segments_anno.json: Instance annotations mapping segment IDs to semantic classes

This script:
1. Loads the semantic mesh and segment annotations
2. For each camera view, renders the mesh to get 2D labels
3. Outputs panoptic PNGs with format: panoptic_id = semantic_id * 10000 + instance_id
   (Uses 10000 multiplier to support ScanNet++ 1000 semantic classes)
4. Generates segments_info JSON for each image

Usage:
    python generate_panoptic_labels.py \
        --scannetpp-root /path/to/scannetpp/data \
        --output-dir /path/to/panoptic_annotations \
        --split-file /path/to/splits/nvs_sem_train.txt \
        --num-workers 8

Or integrated with training:
    python m2f_train_multiview.py \
        --generate-labels \
        --scannetpp-root /path/to/scannetpp/data \
        --panoptic-root /path/to/panoptic_annotations
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# SCANNET++ SCENE HELPER CLASS
# ============================================================

class ScanNetPPScene:
    """Helper class to access ScanNet++ scene data.
    
    Mirrors the structure from scannetpp/common/scene_release.py
    """
    
    def __init__(self, scene_id: str, data_root: str):
        self.scene_id = scene_id
        self.data_root = Path(data_root)
        self.scene_dir = self.data_root / scene_id
    
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
# PLY PARSING UTILITIES
# ============================================================

def read_ply_header(ply_path: str) -> Tuple[int, int, List[str]]:
    """
    Read PLY header to get vertex count and properties.
    
    Returns:
        (num_vertices, header_end_byte, property_names)
    """
    properties = []
    num_vertices = 0
    header_end = 0
    
    with open(ply_path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append(parts[-1])  # Property name
            elif line == 'end_header':
                header_end = f.tell()
                break
    
    return num_vertices, header_end, properties


def load_semantic_mesh_ply(ply_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load semantic PLY mesh with per-vertex labels.
    
    ScanNet++ semantic meshes store labels in vertex colors (RGB).
    The semantic class is encoded in the red channel for the official format,
    or as a separate 'label' or 'semantic' property.
    
    Args:
        ply_path: Path to semantic PLY file
    
    Returns:
        vertices: [N, 3] XYZ coordinates
        labels: [N] semantic class IDs
        faces: [M, 3] face indices (if available)
    """
    try:
        import open3d as o3d
        
        mesh = o3d.io.read_triangle_mesh(ply_path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles) if mesh.has_triangles() else None
        
        # Check for vertex colors (semantic labels encoded in colors)
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            # ScanNet++ encodes semantic ID in the RGB values
            # Usually: label = R (for up to 256 classes)
            # Or label = R + G*256 + B*65536 for more classes
            labels = (colors[:, 0] * 255).astype(np.int32)
        else:
            # No colors, try to read as custom property
            labels = np.zeros(len(vertices), dtype=np.int32)
            logger.warning(f"No vertex colors found in {ply_path}, using zeros")
        
        return vertices, labels, faces
        
    except ImportError:
        logger.error("Open3D not installed. Install with: pip install open3d")
        raise


def load_segments_json(segments_path: str) -> Dict[int, int]:
    """
    Load segments.json which maps vertex indices to segment IDs.
    
    Returns:
        Dict mapping vertex_index -> segment_id
    """
    with open(segments_path, 'r') as f:
        data = json.load(f)
    
    # segments.json format: {"segIndices": [seg_id_for_vertex_0, seg_id_for_vertex_1, ...]}
    seg_indices = data.get('segIndices', [])
    return {i: seg_id for i, seg_id in enumerate(seg_indices)}


def load_segments_anno_json(anno_path: str) -> Dict[int, Dict]:
    """
    Load segments_anno.json which maps segment IDs to semantic/instance info.
    
    Returns:
        Dict mapping segment_id -> {
            'label': semantic_class_name,
            'label_id': semantic_class_id,
            'instance_id': instance_id,
            'objectId': unique object ID
        }
    """
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    segment_to_info = {}
    
    # segments_anno.json format varies, handle both formats
    if 'segGroups' in data:
        # Official ScanNet++ format
        for group in data['segGroups']:
            obj_id = group.get('objectId', group.get('id', 0))
            label = group.get('label', 'unknown')
            label_id = group.get('label_id', 0)
            segments = group.get('segments', [])
            
            for seg_id in segments:
                segment_to_info[seg_id] = {
                    'label': label,
                    'label_id': label_id,
                    'instance_id': obj_id,
                    'objectId': obj_id,
                }
    elif 'annotations' in data:
        # Alternative format
        for anno in data['annotations']:
            seg_id = anno.get('segment_id', anno.get('id'))
            segment_to_info[seg_id] = {
                'label': anno.get('label', 'unknown'),
                'label_id': anno.get('label_id', anno.get('category_id', 0)),
                'instance_id': anno.get('instance_id', anno.get('objectId', 0)),
                'objectId': anno.get('objectId', 0),
            }
    
    return segment_to_info


# ============================================================
# SEMANTIC CLASS MAPPING
# ============================================================

def load_scannetpp_class_mapping(metadata_path: Optional[str] = None) -> Dict[str, int]:
    """
    Load ScanNet++ semantic class name to ID mapping.
    
    If no path provided, uses a default mapping for common classes.
    """
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    # Default mapping for ScanNet++ benchmark top 100 classes
    # This is a subset - full mapping should come from metadata/semantic_classes.txt
    default_mapping = {
        'wall': 1,
        'floor': 2,
        'ceiling': 3,
        'door': 4,
        'window': 5,
        'chair': 6,
        'table': 7,
        'sofa': 8,
        'bed': 9,
        'desk': 10,
        'cabinet': 11,
        'shelf': 12,
        'lamp': 13,
        'curtain': 14,
        'pillow': 15,
        'mirror': 16,
        'picture': 17,
        'sink': 18,
        'toilet': 19,
        'bathtub': 20,
        # Add more as needed...
        'unknown': 0,
        'unlabeled': 255,
    }
    return default_mapping


# Thing vs Stuff classification
THING_CLASSES = {
    'chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf', 'lamp',
    'pillow', 'mirror', 'picture', 'sink', 'toilet', 'bathtub', 'plant',
    'tv', 'monitor', 'computer', 'keyboard', 'mouse', 'phone', 'book',
    'bottle', 'cup', 'bowl', 'box', 'bag', 'shoe', 'clothes',
}

STUFF_CLASSES = {
    'wall', 'floor', 'ceiling', 'door', 'window', 'curtain', 'rug',
    'carpet', 'blinds', 'towel',
}


def is_thing_class(label: str) -> bool:
    """Check if a class is a 'thing' (instance) vs 'stuff' (semantic only)."""
    label_lower = label.lower()
    return label_lower in THING_CLASSES or label_lower not in STUFF_CLASSES


# ============================================================
# MESH RENDERING TO 2D
# ============================================================

def render_mesh_to_2d(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: Dict,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render mesh with per-vertex labels to 2D image.
    
    Uses z-buffering to handle occlusions.
    
    Args:
        vertices: [N, 3] mesh vertices in world coordinates
        faces: [M, 3] face indices
        vertex_labels: [N] per-vertex label IDs
        camera_pose: 4x4 camera-to-world matrix
        intrinsics: Dict with fl_x, fl_y, cx, cy
        image_size: (H, W) output size
    
    Returns:
        label_map: [H, W] per-pixel label IDs
        depth_map: [H, W] per-pixel depth values
    """
    H, W = image_size
    
    # Camera-to-world -> World-to-camera
    world_to_cam = np.linalg.inv(camera_pose)
    
    # Transform vertices to camera coordinates
    verts_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
    verts_cam = (world_to_cam @ verts_homo.T).T[:, :3]
    
    # Project to image coordinates
    fx = intrinsics.get('fl_x', intrinsics.get('fx', 500))
    fy = intrinsics.get('fl_y', intrinsics.get('fy', 500))
    cx = intrinsics.get('cx', W / 2)
    cy = intrinsics.get('cy', H / 2)
    
    # Filter vertices behind camera
    valid_z = verts_cam[:, 2] > 0.01
    
    # Project to 2D
    x_proj = (verts_cam[:, 0] * fx / verts_cam[:, 2] + cx).astype(np.float32)
    y_proj = (verts_cam[:, 1] * fy / verts_cam[:, 2] + cy).astype(np.float32)
    z_proj = verts_cam[:, 2]
    
    # Initialize output buffers
    label_map = np.zeros((H, W), dtype=np.int32)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    
    # Rasterize each triangle
    for face in tqdm(faces, desc="Rasterizing", leave=False, disable=True):
        v0, v1, v2 = face
        
        # Skip if any vertex is behind camera
        if not (valid_z[v0] and valid_z[v1] and valid_z[v2]):
            continue
        
        # Get projected coordinates
        pts = np.array([
            [x_proj[v0], y_proj[v0]],
            [x_proj[v1], y_proj[v1]],
            [x_proj[v2], y_proj[v2]],
        ], dtype=np.float32)
        
        # Get depth values
        depths = np.array([z_proj[v0], z_proj[v1], z_proj[v2]])
        
        # Get labels (use majority vote or first vertex)
        labels = np.array([vertex_labels[v0], vertex_labels[v1], vertex_labels[v2]])
        face_label = int(np.median(labels))  # Majority label
        
        # Bounding box
        min_x = max(0, int(np.floor(pts[:, 0].min())))
        max_x = min(W - 1, int(np.ceil(pts[:, 0].max())))
        min_y = max(0, int(np.floor(pts[:, 1].min())))
        max_y = min(H - 1, int(np.ceil(pts[:, 1].max())))
        
        if min_x >= max_x or min_y >= max_y:
            continue
        
        # Simple triangle rasterization with barycentric coordinates
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Compute barycentric coordinates
                p = np.array([x + 0.5, y + 0.5])
                v0p = pts[0] - p
                v1p = pts[1] - p
                v2p = pts[2] - p
                
                # Cross products for barycentric
                area = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                if abs(area) < 1e-6:
                    continue
                
                w0 = np.cross(v1p, v2p) / area
                w1 = np.cross(v2p, v0p) / area
                w2 = np.cross(v0p, v1p) / area
                
                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate depth
                    z = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
                    
                    # Z-buffer test
                    if z < depth_map[y, x]:
                        depth_map[y, x] = z
                        label_map[y, x] = face_label
    
    return label_map, depth_map


def render_mesh_to_2d_open3d(
    mesh_path: str,
    vertex_panoptic_ids: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: Dict,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render mesh to 2D using Open3D's raycasting (much faster than CPU rasterization).
    
    This uses the same approach as depth rendering but assigns labels based on
    which triangle is hit.
    
    Args:
        mesh_path: Path to mesh PLY file
        vertex_panoptic_ids: [N] panoptic ID per vertex
        camera_pose: 4x4 camera-to-world matrix
        intrinsics: Camera intrinsics dict
        image_size: (H, W) output size
    
    Returns:
        panoptic_map: [H, W] per-pixel panoptic IDs
        depth_map: [H, W] per-pixel depth
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D required for fast mesh rendering. Install with: pip install open3d")
    
    H, W = image_size
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        logger.warning(f"Mesh has no triangles: {mesh_path}")
        return np.zeros((H, W), dtype=np.int32), np.zeros((H, W), dtype=np.float32)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # Camera parameters
    fx = intrinsics.get('fl_x', intrinsics.get('fx', 500))
    fy = intrinsics.get('fl_y', intrinsics.get('fy', 500))
    cx = intrinsics.get('cx', W / 2)
    cy = intrinsics.get('cy', H / 2)
    
    # Camera-to-world matrix
    c2w = camera_pose
    
    # Generate rays for each pixel
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    # Pixel to normalized camera coordinates
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam)
    
    # Ray directions in camera frame
    rays_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
    
    # Transform to world frame
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    rays_world = (R @ rays_cam.T).T
    origins = np.tile(t, (len(rays_cam), 1))
    
    # Create ray tensor for Open3D
    rays = np.hstack([origins, rays_world]).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    # Cast rays
    result = scene.cast_rays(rays_tensor)
    
    # Extract hit information
    t_hit = result['t_hit'].numpy()
    primitive_ids = result['primitive_ids'].numpy()
    
    # Reshape to image
    depth_map = t_hit.reshape(H, W)
    triangle_ids = primitive_ids.reshape(H, W)
    
    # Map triangle hits to panoptic IDs
    panoptic_map = np.zeros((H, W), dtype=np.int32)
    
    # Check for valid hits: t_hit < inf and primitive_id is valid
    # Open3D returns UINT32_MAX (4294967295) for misses, so check against that
    INVALID_ID = 4294967295  # UINT32_MAX, returned by Open3D for ray misses
    valid_hits = (triangle_ids != INVALID_ID) & (np.isfinite(depth_map)) & (depth_map > 0)
    
    num_triangles = len(triangles)
    
    # Vectorized approach: process all valid hits at once
    valid_y, valid_x = np.where(valid_hits)
    if len(valid_y) > 0:
        valid_tri_ids = triangle_ids[valid_y, valid_x].astype(np.int64)
        
        # Filter out any out-of-bounds triangle IDs
        in_bounds = (valid_tri_ids >= 0) & (valid_tri_ids < num_triangles)
        valid_y = valid_y[in_bounds]
        valid_x = valid_x[in_bounds]
        valid_tri_ids = valid_tri_ids[in_bounds]
        
        if len(valid_tri_ids) > 0:
            # Get vertex indices for all valid triangles
            v0_ids = triangles[valid_tri_ids, 0]
            v1_ids = triangles[valid_tri_ids, 1]
            v2_ids = triangles[valid_tri_ids, 2]
            
            # Get panoptic IDs for each vertex
            labels_v0 = vertex_panoptic_ids[v0_ids]
            labels_v1 = vertex_panoptic_ids[v1_ids]
            labels_v2 = vertex_panoptic_ids[v2_ids]
            
            # Use median (majority vote) of the three vertex labels
            all_labels = np.stack([labels_v0, labels_v1, labels_v2], axis=1)
            median_labels = np.median(all_labels, axis=1).astype(np.int32)
            
            # Assign to panoptic map
            panoptic_map[valid_y, valid_x] = median_labels
    
    # Handle misses (infinite depth)
    depth_map[~valid_hits] = 0
    
    return panoptic_map, depth_map


# ============================================================
# PANOPTIC LABEL GENERATION
# ============================================================

def generate_panoptic_for_scene(
    scene_id: str,
    data_root: str,
    output_dir: str,
    image_type: str = 'dslr',
    use_undistorted: bool = True,
    class_mapping: Optional[Dict[str, int]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Generate 2D panoptic labels for all images in a scene.
    
    Args:
        scene_id: ScanNet++ scene identifier
        data_root: Root directory of ScanNet++ data
        output_dir: Output directory for panoptic annotations
        image_type: 'dslr' or 'iphone'
        use_undistorted: Use undistorted images
        class_mapping: Optional semantic class name to ID mapping
        skip_existing: Skip scenes that already have annotations
    
    Returns:
        Dict with generation statistics
    """
    # ScanNetPPScene and read_nerfstudio_transforms are now defined in this file
    scene = ScanNetPPScene(scene_id, data_root)
    scene_output_dir = Path(output_dir) / scene_id
    
    # Check if already processed
    if skip_existing and scene_output_dir.exists():
        existing_files = list(scene_output_dir.glob('*.png'))
        if len(existing_files) > 0:
            logger.info(f"Skipping {scene_id}: {len(existing_files)} annotations already exist")
            return {'scene_id': scene_id, 'status': 'skipped', 'num_images': len(existing_files)}
    
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mesh and annotations
    semantic_mesh_path = scene.semantic_mesh_path
    segments_path = scene.segments_path
    segments_anno_path = scene.segments_anno_path
    
    if not semantic_mesh_path.exists():
        logger.warning(f"Semantic mesh not found: {semantic_mesh_path}")
        # Fall back to regular mesh
        mesh_path = scene.mesh_path
        if not mesh_path.exists():
            logger.error(f"No mesh found for scene {scene_id}")
            return {'scene_id': scene_id, 'status': 'error', 'error': 'no mesh'}
        semantic_mesh_path = mesh_path
    
    # Load mesh vertices and labels
    try:
        vertices, vertex_labels, faces = load_semantic_mesh_ply(str(semantic_mesh_path))
    except Exception as e:
        logger.error(f"Failed to load mesh for {scene_id}: {e}")
        return {'scene_id': scene_id, 'status': 'error', 'error': str(e)}
    
    # Load segment annotations for instance info
    segment_to_vertex = {}
    segment_to_info = {}
    
    if segments_path.exists():
        segment_to_vertex = load_segments_json(str(segments_path))
    
    if segments_anno_path.exists():
        segment_to_info = load_segments_anno_json(str(segments_anno_path))
    
    # Build vertex -> panoptic_id mapping
    # panoptic_id = semantic_id * 10000 + instance_id
    # Uses 10000 multiplier to support ScanNet++ 1000 semantic classes
    vertex_panoptic_ids = np.zeros(len(vertices), dtype=np.int32)
    
    if class_mapping is None:
        class_mapping = load_scannetpp_class_mapping()
    
    # Track unique segments for segments_info
    all_segments_info = {}
    
    for v_idx in range(len(vertices)):
        seg_id = segment_to_vertex.get(v_idx, -1)
        
        if seg_id in segment_to_info:
            info = segment_to_info[seg_id]
            label_name = info.get('label', 'unknown').lower()
            semantic_id = class_mapping.get(label_name, info.get('label_id', 0))
            instance_id = info.get('instance_id', 0)
            
            # For stuff classes, instance_id = 0
            if not is_thing_class(label_name):
                instance_id = 0
            
            panoptic_id = semantic_id * 10000 + (instance_id % 10000)
            vertex_panoptic_ids[v_idx] = panoptic_id
            
            # Store segment info
            if panoptic_id not in all_segments_info:
                all_segments_info[panoptic_id] = {
                    'id': int(panoptic_id),
                    'category_id': int(semantic_id),
                    'isthing': bool(is_thing_class(label_name)),
                    'area': 0,  # Will be computed per-image
                }
        else:
            # Use vertex color label if no segment annotation
            semantic_id = int(vertex_labels[v_idx])
            vertex_panoptic_ids[v_idx] = semantic_id * 10000
    
    # Load camera data
    if image_type == 'dslr':
        transforms_path = (
            scene.dslr_nerfstudio_transforms_undistorted if use_undistorted
            else scene.dslr_nerfstudio_transforms
        )
        image_dir = (
            scene.dslr_undistorted_dir if use_undistorted
            else scene.dslr_resized_dir
        )
    else:
        transforms_path = scene.iphone_nerfstudio_transforms
        image_dir = scene.iphone_rgb_dir
    
    if not transforms_path.exists():
        logger.error(f"Transforms not found: {transforms_path}")
        return {'scene_id': scene_id, 'status': 'error', 'error': 'no transforms'}
    
    intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
    
    # Determine image size
    image_w = intrinsics.get('w', 640)
    image_h = intrinsics.get('h', 480)
    
    # Generate panoptic for each frame
    num_generated = 0
    
    for frame in tqdm(frames, desc=f"Scene {scene_id}", leave=False):
        file_path = frame.get('file_path', '')
        image_name = Path(file_path).stem
        
        output_png = scene_output_dir / f"{image_name}.png"
        output_json = scene_output_dir / f"{image_name}.json"
        
        if skip_existing and output_png.exists():
            num_generated += 1
            continue
        
        # Get camera pose
        c2w = frame.get('camera_to_world')
        if c2w is None:
            if 'transform_matrix' in frame:
                c2w = np.array(frame['transform_matrix'])
            else:
                logger.warning(f"No pose for {image_name}")
                continue
        
        if isinstance(c2w, list):
            c2w = np.array(c2w)
        
        # Render panoptic map
        try:
            panoptic_map, depth_map = render_mesh_to_2d_open3d(
                str(scene.mesh_path),
                vertex_panoptic_ids,
                c2w,
                intrinsics,
                (image_h, image_w),
            )
        except Exception as e:
            logger.warning(f"Rendering failed for {image_name}: {e}")
            continue
        
        # Encode panoptic map as RGB PNG
        # panoptic_id = R + G*256 + B*256*256
        # NOTE: cv2.imwrite expects BGR input, so the [:,:,::-1] flip means the
        # on-disk byte order is actually B,G,R (i.e., the least significant byte
        # of the ID ends up in what PIL reads as the B channel). The dataset mapper
        # compensates for this by using B + G*256 + R*65536 when decoding.
        panoptic_rgb = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        panoptic_rgb[:, :, 0] = panoptic_map % 256
        panoptic_rgb[:, :, 1] = (panoptic_map // 256) % 256
        panoptic_rgb[:, :, 2] = (panoptic_map // 65536) % 256
        
        cv2.imwrite(str(output_png), panoptic_rgb[:, :, ::-1])  # RGB to BGR for cv2
        
        # Compute per-image segments_info
        unique_ids, counts = np.unique(panoptic_map, return_counts=True)
        image_segments_info = []
        
        for pan_id, area in zip(unique_ids, counts):
            if pan_id == 0:
                continue  # Skip background
            
            semantic_id = pan_id // 10000
            instance_id = pan_id % 10000
            
            # Check if thing or stuff (convert to native Python bool for JSON)
            is_thing = bool(instance_id > 0)
            
            image_segments_info.append({
                'id': int(pan_id),
                'category_id': int(semantic_id),
                'isthing': is_thing,
                'area': int(area),
            })
        
        # Save segments_info JSON
        with open(output_json, 'w') as f:
            json.dump({'segments_info': image_segments_info}, f, indent=2)
        
        num_generated += 1
    
    return {
        'scene_id': scene_id,
        'status': 'success',
        'num_images': num_generated,
        'num_segments': len(all_segments_info),
    }


def generate_panoptic_labels_parallel(
    data_root: str,
    output_dir: str,
    split_file: str,
    image_type: str = 'dslr',
    use_undistorted: bool = True,
    num_workers: int = 4,
    skip_existing: bool = True,
):
    """
    Generate panoptic labels for all scenes in a split file.
    
    Args:
        data_root: ScanNet++ data root
        output_dir: Output directory for annotations
        split_file: Path to split file (list of scene IDs)
        image_type: 'dslr' or 'iphone'
        use_undistorted: Use undistorted images
        num_workers: Number of parallel workers
        skip_existing: Skip scenes with existing annotations
    """
    # Read scene IDs from split file
    with open(split_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Generating panoptic labels for {len(scene_ids)} scenes")
    logger.info(f"Output directory: {output_dir}")
    
    # Load class mapping once
    class_mapping = load_scannetpp_class_mapping()
    
    results = []
    
    if num_workers <= 1:
        # Sequential processing
        for scene_id in tqdm(scene_ids, desc="Generating labels"):
            result = generate_panoptic_for_scene(
                scene_id=scene_id,
                data_root=data_root,
                output_dir=output_dir,
                image_type=image_type,
                use_undistorted=use_undistorted,
                class_mapping=class_mapping,
                skip_existing=skip_existing,
            )
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    generate_panoptic_for_scene,
                    scene_id=scene_id,
                    data_root=data_root,
                    output_dir=output_dir,
                    image_type=image_type,
                    use_undistorted=use_undistorted,
                    class_mapping=class_mapping,
                    skip_existing=skip_existing,
                ): scene_id
                for scene_id in scene_ids
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating labels"):
                scene_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {scene_id}: {e}")
                    results.append({'scene_id': scene_id, 'status': 'error', 'error': str(e)})
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    total_images = sum(r.get('num_images', 0) for r in results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Panoptic Label Generation Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Success: {success} scenes")
    logger.info(f"Skipped: {skipped} scenes")
    logger.info(f"Errors:  {errors} scenes")
    logger.info(f"Total images: {total_images}")
    
    # Save summary
    summary_path = Path(output_dir) / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'total_scenes': len(scene_ids),
            'success': success,
            'skipped': skipped,
            'errors': errors,
            'total_images': total_images,
            'results': results,
        }, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")


# ============================================================
# MAIN (Standalone Usage)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D panoptic labels from ScanNet++ 3D mesh annotations"
    )
    
    parser.add_argument(
        "--scannetpp-root",
        type=str,
        required=True,
        help="Path to ScanNet++ data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for panoptic annotations"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to split file (list of scene IDs)"
    )
    parser.add_argument(
        "--image-type",
        type=str,
        default="dslr",
        choices=["dslr", "iphone"],
        help="Image type to process"
    )
    parser.add_argument(
        "--use-undistorted",
        action="store_true",
        default=True,
        help="Use undistorted images"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip scenes with existing annotations"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Process single scene only (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.scene_id:
        # Process single scene
        result = generate_panoptic_for_scene(
            scene_id=args.scene_id,
            data_root=args.scannetpp_root,
            output_dir=args.output_dir,
            image_type=args.image_type,
            use_undistorted=args.use_undistorted,
            skip_existing=args.skip_existing,
        )
        print(f"Result: {result}")
    else:
        # Process all scenes in split
        generate_panoptic_labels_parallel(
            data_root=args.scannetpp_root,
            output_dir=args.output_dir,
            split_file=args.split_file,
            image_type=args.image_type,
            use_undistorted=args.use_undistorted,
            num_workers=args.num_workers,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
