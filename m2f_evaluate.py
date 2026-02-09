#!/usr/bin/env python3
"""
Standalone Evaluation Script for Multi-View Mask2Former on ScanNet++

Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and
Recognition Quality (RQ) on the validation set using per-image
ground truth annotations (panoptic_val/<scene_id>/<image_stem>.json).

This script uses the official `panopticapi.evaluation.pq_compute` function
(the same one used internally by Detectron2's COCOPanopticEvaluator) to
compute PQ/SQ/RQ.  Because the ScanNet++ dataset stores per-image GT
JSON+PNG files instead of a single merged COCO-format JSON, this script:

1. Loads the trained model checkpoint.
2. Iterates over every validation image (one view at a time).
3. Runs single-view panoptic inference through the trained model.
4. Saves predicted panoptic PNGs + assembles a COCO-format prediction JSON.
5. Assembles a COCO-format GT JSON from the per-image GT annotations.
6. Calls panopticapi.evaluation.pq_compute to compute official metrics.
7. Reports overall PQ, SQ, RQ as well as things/stuff splits.

Usage (single GPU):
    python m2f_evaluate.py \
        --config-file configs/scannetpp/panoptic-segmentation/ma40.yaml \
        --checkpoint /path/to/model_final.pth \
        --output-dir ./eval_results \
        MODEL.WEIGHTS /path/to/model_final.pth

Usage via SLURM:
    sbatch eval_job.sh
"""

import os
import sys
import json
import copy
import time
import logging
import warnings
import argparse
import csv
import contextlib
import io as _io
import tempfile
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Detectron2
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
import detectron2.utils.comm as comm

from mask2former import add_maskformer2_config

# Official panoptic evaluation API (same as used by Detectron2 internally)
from panopticapi.utils import id2rgb, rgb2id
from panopticapi.evaluation import pq_compute

# Import model & config helpers from training script
from m2f_train_multiview_working import (
    add_multiview_config,
    setup_cfg,
    MultiViewMask2Former,
    MapAnythingMultiViewBackbone,
    multi_view_collate_fn,
)

# Import dataset utilities
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    ScanNetPPScene,
    read_nerfstudio_transforms,
    read_colmap_cameras,
    read_colmap_images,
    build_intrinsic_matrix,
    _load_scannetpp_panoptic_dataset,
)

# Import registration
from register_datasets import register_scannetpp_train_val

# Filter FutureWarnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

logger = logging.getLogger("m2f_evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ============================================================
# CONSTANTS
# ============================================================
LABEL_DIVISOR = 10000      # panoptic_id = semantic_id * 10000 + instance_id
IGNORE_LABEL = 255
VOID_PANOPTIC_ID = 0       # ID 0 treated as void/unlabeled


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(cfg, checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = MultiViewMask2Former(cfg)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    return model


# ============================================================
# SINGLE-VIEW INFERENCE
# ============================================================

@torch.no_grad()
def run_single_view_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    camera_pose: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    num_classes: int,
    overlap_threshold: float = 0.8,
    object_mask_threshold: float = 0.8,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run panoptic inference on a single image.

    The model expects multi-view input, so we wrap a single view into
    the expected batch format (B=1, N=1).

    Args:
        model: Trained MultiViewMask2Former
        image: [3, H, W] float tensor (already normalised to 0-255 range)
        camera_pose: [4, 4] camera-to-world
        camera_intrinsic: [3, 3] intrinsic matrix
        num_classes: number of semantic classes
        overlap_threshold: panoptic post-processing overlap threshold
        object_mask_threshold: panoptic post-processing object threshold

    Returns:
        panoptic_map: [H, W] int32 with panoptic IDs
        segments_info: list of dicts with id, category_id, isthing, area
    """
    device = next(model.parameters()).device

    # Wrap as multi-view batch: B=1, N=1
    B, N = 1, 1
    C, H, W = image.shape
    images = image.unsqueeze(0).unsqueeze(0).to(device)            # [1,1,3,H,W]
    poses = camera_pose.unsqueeze(0).unsqueeze(0).to(device)       # [1,1,4,4]
    intrinsics = camera_intrinsic.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,3]

    # Build a dummy batch dict that looks like multi_view_collate_fn output
    batch = {
        "images": images,
        "camera_poses": poses,
        "camera_intrinsics": intrinsics,
        "scene_ids": ["eval"],
        "file_names": ["eval"],
        "instances": [None],
    }

    # Run the backbone to get features for view 0
    backbone_out = model.backbone(
        images=images,
        camera_poses=poses,
        camera_intrinsics=intrinsics,
        return_all_views=True,
    )
    all_view_features = backbone_out["all_view_features"]

    # Extract view 0 features
    view_features = {}
    for key in ["res2", "res3", "res4", "res5"]:
        view_features[key] = all_view_features[key][0]  # [B, C, H', W']

    # Run pixel decoder
    mask_features, multi_scale_features = model._run_pixel_decoder(view_features)

    # Run transformer decoder (no query propagation)
    outputs = model.query_propagation_decoder(
        multi_scale_features,
        mask_features,
        mask=None,
        initial_query_feat=None,
        initial_attn_mask=None,
    )

    pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
    pred_masks = outputs["pred_masks"]    # [B, Q, H', W']

    # Upsample masks to original resolution
    pred_masks = F.interpolate(
        pred_masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    # Panoptic post-processing (following Mask2Former)
    panoptic_map, segments_info = panoptic_postprocessing(
        pred_logits[0],
        pred_masks[0],
        num_classes=num_classes,
        overlap_threshold=overlap_threshold,
        object_mask_threshold=object_mask_threshold,
    )

    return panoptic_map, segments_info


def panoptic_postprocessing(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    num_classes: int,
    overlap_threshold: float = 0.8,
    object_mask_threshold: float = 0.8,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert per-query logits + masks to a panoptic segmentation map.

    Follows the standard Mask2Former panoptic post-processing.

    Args:
        pred_logits: [Q, C+1] class logits (last class = no-object)
        pred_masks: [Q, H, W] mask logits (pre-sigmoid)
        num_classes: number of valid classes (excluding no-object)
        overlap_threshold: IoU threshold for merging overlapping masks
        object_mask_threshold: confidence threshold for mask binarisation

    Returns:
        panoptic_map: [H, W] int32 panoptic ID map
        segments_info: list of segment dicts
    """
    Q, H, W = pred_masks.shape

    # Class probabilities (exclude no-object class)
    scores = F.softmax(pred_logits, dim=-1)  # [Q, C+1]
    scores = scores[:, :num_classes]          # [Q, C]
    max_scores, labels = scores.max(dim=-1)  # [Q], [Q]

    # Mask probabilities
    mask_probs = pred_masks.sigmoid()  # [Q, H, W]

    # Combined score: class_score * mask_confidence
    mask_scores = (mask_probs > 0.5).float().sum(dim=(1, 2))  # per-query area
    combined_scores = max_scores

    # Sort by score (descending)
    sorted_indices = torch.argsort(combined_scores, descending=True)

    # Greedy panoptic merging
    panoptic_map = torch.zeros(H, W, dtype=torch.int32, device=pred_masks.device)
    segments_info = []
    current_segment_id = 0
    occupied = torch.zeros(H, W, dtype=torch.bool, device=pred_masks.device)

    for idx in sorted_indices:
        score = combined_scores[idx].item()
        if score < 0.05:  # very low confidence → skip
            continue

        cat_id = labels[idx].item()
        binary_mask = mask_probs[idx] > 0.5  # [H, W]

        # Check overlap with already placed segments
        overlap = (binary_mask & occupied).sum().item()
        mask_area = binary_mask.sum().item()

        if mask_area == 0:
            continue

        if overlap / mask_area > overlap_threshold:
            continue  # too much overlap

        # Remove overlapping pixels
        binary_mask = binary_mask & (~occupied)
        mask_area = binary_mask.sum().item()

        if mask_area == 0:
            continue

        # Assign panoptic ID: category_id * LABEL_DIVISOR + instance_id
        current_segment_id += 1
        panoptic_id = cat_id * LABEL_DIVISOR + current_segment_id

        panoptic_map[binary_mask] = panoptic_id
        occupied |= binary_mask

        segments_info.append({
            "id": panoptic_id,
            "category_id": cat_id,
            "isthing": 1 if score > 0 else 0,  # will refine below
            "area": mask_area,
            "score": score,
        })

    return panoptic_map.cpu().numpy(), segments_info


# ============================================================
# GROUND TRUTH LOADING
# ============================================================

def load_gt_panoptic(
    panoptic_png_path: str,
    segments_info_path: str,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load ground-truth panoptic map and segments_info.

    The panoptic PNG was saved with cv2.imwrite(path, rgb[:,:,::-1])
    which swaps R↔B.  When reading as RGB the channel order is swapped.
    Encoding: panoptic_id = B + G*256 + R*65536  (after RGB read).

    Returns:
        panoptic_map: [H, W] int32 decoded panoptic ID map
        segments_info: list of segment dicts from the JSON
    """
    # Load segments info JSON
    with open(segments_info_path, "r") as f:
        segments_info = json.load(f)

    # Load panoptic PNG as RGB
    pan_rgb = np.array(Image.open(panoptic_png_path).convert("RGB"))

    # Decode panoptic ID (accounting for BGR swap during save)
    panoptic_map = (
        pan_rgb[:, :, 2].astype(np.int32)           # B channel = LSB
        + pan_rgb[:, :, 1].astype(np.int32) * 256   # G channel = middle
        + pan_rgb[:, :, 0].astype(np.int32) * 65536  # R channel = MSB
    )

    return panoptic_map, segments_info


def re_encode_panoptic_png(panoptic_map: np.ndarray) -> Image.Image:
    """
    Re-encode a decoded panoptic ID map into an RGB PNG using the
    standard panopticapi id2rgb encoding so that `rgb2id` will
    correctly recover the original IDs.

    Args:
        panoptic_map: [H, W] int32 with decoded panoptic IDs

    Returns:
        PIL Image in RGB with panopticapi-compatible encoding
    """
    return Image.fromarray(id2rgb(panoptic_map))


# ============================================================
# IMAGE LOADING AND PREPARATION
# ============================================================

def load_and_prepare_image(
    image_path: str,
    target_short_edge: int = 480,
    max_size: int = 640,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load an image, resize, and return as a float tensor.

    Returns:
        image: [3, H, W] float32 tensor (0-255 range, RGB)
        original_size: (orig_H, orig_W)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Resize shortest edge
    scale = target_short_edge / min(orig_h, orig_w)
    if max(orig_h, orig_w) * scale > max_size:
        scale = max_size / max(orig_h, orig_w)

    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to be divisible by 32
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                     mode="constant", constant_values=128)

    # To tensor [3, H, W] float32
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img_tensor, (orig_h, orig_w)


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def build_categories_list(num_classes: int, metadata=None) -> List[Dict]:
    """
    Build the COCO-format 'categories' list needed by panopticapi.

    Each entry: {"id": <int>, "name": <str>, "isthing": <0 or 1>}

    If metadata (from MetadataCatalog) is available and has thing/stuff
    ID mappings, they are used.  Otherwise all classes are treated as stuff.
    """
    categories = []
    thing_ids = set()
    if metadata is not None:
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            thing_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())

    for cid in range(num_classes):
        categories.append({
            "id": cid,
            "name": str(cid),
            "isthing": 1 if cid in thing_ids else 0,
        })
    return categories


def evaluate(
    model: torch.nn.Module,
    scannetpp_root: str,
    panoptic_val_root: str,
    val_split_file: str,
    output_dir: str,
    num_classes: int = 254,
    target_short_edge: int = 480,
    max_size: int = 640,
    overlap_threshold: float = 0.8,
    object_mask_threshold: float = 0.8,
    max_images_per_scene: Optional[int] = None,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation over the entire validation set using panopticapi.

    For each image:
      1. Run inference to get a predicted panoptic map + segments_info.
      2. Load the per-image GT panoptic PNG + JSON.
      3. Re-encode both GT and pred panoptic maps into panopticapi-compatible
         RGB PNGs (using id2rgb) so that pq_compute can decode them with rgb2id.
      4. Accumulate COCO-format annotation entries for GT and pred.

    After all images are processed, call panopticapi.evaluation.pq_compute
    with the assembled GT JSON, prediction JSON, and the PNG folders.

    Args:
        model: Trained model (already on GPU)
        scannetpp_root: Path to ScanNet++ data root
        panoptic_val_root: Path to panoptic_val directory with GT JSONs + PNGs
        val_split_file: Path to validation split file
        output_dir: Directory for output CSVs / predictions
        num_classes: Number of semantic classes
        target_short_edge: Resize short edge for inference
        max_size: Maximum image dimension
        overlap_threshold: Panoptic post-processing overlap threshold
        object_mask_threshold: Mask binarisation threshold
        max_images_per_scene: Limit images per scene (None = all)
        save_predictions: Save predicted panoptic PNGs to output dir

    Returns:
        Dict with PQ, SQ, RQ metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read validation scenes
    with open(val_split_file, "r") as f:
        val_scenes = [line.strip() for line in f if line.strip()]

    logger.info(f"Evaluating on {len(val_scenes)} validation scenes")
    logger.info(f"GT panoptic root: {panoptic_val_root}")
    logger.info(f"Image short edge: {target_short_edge}, max size: {max_size}")

    model.eval()
    device = next(model.parameters()).device

    # Try to get metadata for thing/stuff class info
    metadata = None
    try:
        metadata = MetadataCatalog.get("scannetpp_panoptic_val")
    except Exception:
        try:
            metadata = MetadataCatalog.get("scannetpp_panoptic_train")
        except Exception:
            pass

    categories = build_categories_list(num_classes, metadata)
    categories_dict = {c["id"]: c for c in categories}

    # Temp directories for panopticapi-compatible PNGs
    gt_png_dir = os.path.join(output_dir, "gt_panoptic_pngs")
    pred_png_dir = os.path.join(output_dir, "pred_panoptic_pngs")
    os.makedirs(gt_png_dir, exist_ok=True)
    os.makedirs(pred_png_dir, exist_ok=True)

    # COCO-format annotation lists
    gt_annotations = []
    pred_annotations = []
    images_list = []

    total_images = 0
    skipped_images = 0
    image_id_counter = 0

    for scene_id in tqdm(val_scenes, desc="Scenes"):
        scene = ScanNetPPScene(scene_id, scannetpp_root)
        scene_panoptic_dir = Path(panoptic_val_root) / scene_id

        if not scene_panoptic_dir.exists():
            logger.warning(f"No panoptic GT for scene {scene_id}, skipping")
            continue

        # Load camera data — try NerfStudio transforms first, COLMAP as fallback
        intrinsics = None
        frames = []

        transforms_path = scene.dslr_nerfstudio_transforms_undistorted
        if not transforms_path.exists():
            transforms_path = scene.dslr_nerfstudio_transforms

        if transforms_path.exists():
            intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
        else:
            # COLMAP fallback (same as ScanNetPPMultiViewDatasetMapper._load_scene_data)
            colmap_dir = scene.dslr_colmap_dir
            cameras_file = colmap_dir / 'cameras.txt'
            images_file = colmap_dir / 'images.txt'

            if cameras_file.exists() and images_file.exists():
                cameras = read_colmap_cameras(str(cameras_file))
                colmap_images = read_colmap_images(str(images_file))

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

                # Get image directory for constructing frame file paths
                image_dir_for_colmap = scene.dslr_undistorted_dir
                if not image_dir_for_colmap.exists():
                    image_dir_for_colmap = scene.dslr_resized_dir

                frames = []
                for name, data in colmap_images.items():
                    frames.append({
                        'file_path': str(image_dir_for_colmap / name),
                        'camera_to_world': data['camera_to_world'],
                        'is_bad': False,
                    })
                logger.info(f"  Scene {scene_id}: loaded {len(frames)} frames from COLMAP")
            else:
                logger.warning(
                    f"No transforms for scene {scene_id}: "
                    f"no NerfStudio at {transforms_path}, "
                    f"no COLMAP at {colmap_dir}"
                )
                continue

        if not frames:
            logger.warning(f"No frames for scene {scene_id}, skipping")
            continue

        K = build_intrinsic_matrix(intrinsics)
        K_tensor = torch.from_numpy(K).float()

        # Build frame lookup
        frame_lookup = {Path(f["file_path"]).stem: f for f in frames}

        # Get image directory
        image_dir = scene.dslr_undistorted_dir
        if not image_dir.exists():
            image_dir = scene.dslr_resized_dir

        # List GT annotation files
        gt_json_files = sorted(scene_panoptic_dir.glob("*.json"))
        if max_images_per_scene is not None:
            gt_json_files = gt_json_files[:max_images_per_scene]

        for json_path in gt_json_files:
            image_stem = json_path.stem  # e.g. "DSC00001"

            # Find the corresponding GT PNG
            gt_png_path = scene_panoptic_dir / f"{image_stem}.png"
            if not gt_png_path.exists():
                skipped_images += 1
                continue

            # Find image file
            image_path = None
            for ext in [".JPG", ".jpg", ".png"]:
                candidate = image_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = str(candidate)
                    break

            if image_path is None:
                skipped_images += 1
                continue

            # Get camera pose
            frame_info = frame_lookup.get(image_stem)
            if frame_info is None:
                for key in frame_lookup:
                    if key.startswith(image_stem):
                        frame_info = frame_lookup[key]
                        break

            if frame_info is None:
                skipped_images += 1
                continue

            c2w = frame_info["camera_to_world"]
            if isinstance(c2w, np.ndarray):
                c2w = torch.from_numpy(c2w).float()
            else:
                c2w = torch.tensor(c2w).float()

            try:
                # ------- Load and decode GT -------
                gt_panoptic_map, gt_segments = load_gt_panoptic(
                    str(gt_png_path), str(json_path)
                )

                # ------- Run inference -------
                img_tensor, orig_size = load_and_prepare_image(
                    image_path, target_short_edge, max_size
                )

                # Scale intrinsics for the resized image
                orig_h, orig_w = orig_size
                _, resized_h, resized_w = img_tensor.shape
                scale_x = resized_w / orig_w
                scale_y = resized_h / orig_h
                K_scaled = K_tensor.clone()
                K_scaled[0, 0] *= scale_x
                K_scaled[1, 1] *= scale_y
                K_scaled[0, 2] *= scale_x
                K_scaled[1, 2] *= scale_y

                pred_panoptic, pred_segments = run_single_view_inference(
                    model=model,
                    image=img_tensor,
                    camera_pose=c2w,
                    camera_intrinsic=K_scaled,
                    num_classes=num_classes,
                    overlap_threshold=overlap_threshold,
                    object_mask_threshold=object_mask_threshold,
                )

                # Resize prediction to match GT resolution
                gt_h, gt_w = gt_panoptic_map.shape
                if pred_panoptic.shape != gt_panoptic_map.shape:
                    pred_panoptic = cv2.resize(
                        pred_panoptic.astype(np.float32),
                        (gt_w, gt_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)

                # ------- Save GT PNG (re-encoded for panopticapi) -------
                image_id_counter += 1
                img_id = image_id_counter
                gt_file_name = f"{scene_id}_{image_stem}.png"
                gt_png_out = re_encode_panoptic_png(gt_panoptic_map)
                gt_png_out.save(os.path.join(gt_png_dir, gt_file_name))

                # Ensure each GT segment has required fields for panopticapi
                gt_ann_segments = []
                for seg in gt_segments:
                    seg_entry = {
                        "id": int(seg["id"]),
                        "category_id": int(seg["category_id"]),
                        "iscrowd": int(seg.get("iscrowd", 0)),
                    }
                    # Compute area from the actual map
                    seg_entry["area"] = int((gt_panoptic_map == seg["id"]).sum())
                    gt_ann_segments.append(seg_entry)

                gt_annotations.append({
                    "image_id": img_id,
                    "file_name": gt_file_name,
                    "segments_info": gt_ann_segments,
                })

                # ------- Save pred PNG (encoded for panopticapi) -------
                pred_file_name = f"{scene_id}_{image_stem}.png"
                pred_png_out = re_encode_panoptic_png(pred_panoptic)
                pred_png_out.save(os.path.join(pred_png_dir, pred_file_name))

                # Determine isthing for predicted segments using categories
                pred_ann_segments = []
                for seg in pred_segments:
                    cat_id = int(seg["category_id"])
                    seg_entry = {
                        "id": int(seg["id"]),
                        "category_id": cat_id,
                    }
                    # Compute area from the actual map
                    seg_entry["area"] = int((pred_panoptic == seg["id"]).sum())
                    pred_ann_segments.append(seg_entry)

                pred_annotations.append({
                    "image_id": img_id,
                    "file_name": pred_file_name,
                    "segments_info": pred_ann_segments,
                })

                images_list.append({
                    "id": img_id,
                    "file_name": f"{scene_id}_{image_stem}",
                })

                total_images += 1

                if total_images % 50 == 0:
                    logger.info(f"  Processed {total_images} images...")

            except Exception as e:
                logger.warning(f"Error processing {scene_id}/{image_stem}: {e}")
                import traceback
                traceback.print_exc()
                skipped_images += 1
                continue

    logger.info(f"Inference complete: {total_images} images, {skipped_images} skipped")

    if total_images == 0:
        logger.error("No images were successfully processed!")
        return {}

    # ---------------------------------------------------------------
    # Build COCO-format JSONs and call panopticapi.evaluation.pq_compute
    # ---------------------------------------------------------------
    logger.info("Assembling COCO-format JSONs for panopticapi evaluation...")

    gt_json_data = {
        "images": images_list,
        "annotations": gt_annotations,
        "categories": categories,
    }
    gt_json_path = os.path.join(output_dir, "gt_panoptic.json")
    with open(gt_json_path, "w") as f:
        json.dump(gt_json_data, f)

    pred_json_data = {
        "images": images_list,
        "annotations": pred_annotations,
        "categories": categories,
    }
    pred_json_path = os.path.join(output_dir, "pred_panoptic.json")
    with open(pred_json_path, "w") as f:
        json.dump(pred_json_data, f)

    logger.info(f"GT JSON: {gt_json_path}  ({len(gt_annotations)} annotations)")
    logger.info(f"Pred JSON: {pred_json_path}  ({len(pred_annotations)} annotations)")
    logger.info(f"GT PNGs: {gt_png_dir}")
    logger.info(f"Pred PNGs: {pred_png_dir}")

    # Run official panopticapi evaluation
    logger.info("Running panopticapi.evaluation.pq_compute ...")
    pq_res = pq_compute(
        gt_json_path,
        pred_json_path,
        gt_folder=gt_png_dir,
        pred_folder=pred_png_dir,
    )

    # Extract results (same format as Detectron2 COCOPanopticEvaluator)
    results = OrderedDict()
    results["PQ"] = 100 * pq_res["All"]["pq"]
    results["SQ"] = 100 * pq_res["All"]["sq"]
    results["RQ"] = 100 * pq_res["All"]["rq"]
    results["PQ_th"] = 100 * pq_res["Things"]["pq"]
    results["SQ_th"] = 100 * pq_res["Things"]["sq"]
    results["RQ_th"] = 100 * pq_res["Things"]["rq"]
    results["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
    results["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
    results["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
    results["n_classes_all"] = pq_res["All"]["n"]
    results["n_classes_things"] = pq_res["Things"]["n"]
    results["n_classes_stuff"] = pq_res["Stuff"]["n"]
    results["num_images"] = total_images

    # Also extract per-class results if available
    per_class_results = {}
    if "per_class" in pq_res:
        for cat_id, metrics in pq_res["per_class"].items():
            per_class_results[int(cat_id)] = {
                "PQ": metrics["pq"] * 100,
                "SQ": metrics["sq"] * 100,
                "RQ": metrics["rq"] * 100,
            }
    results["per_class"] = per_class_results

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS (panopticapi)")
    logger.info("=" * 70)
    logger.info(f"Total images evaluated: {total_images}")
    logger.info(f"Skipped images: {skipped_images}")
    logger.info(f"Active classes (all): {results['n_classes_all']}")
    logger.info(f"Active classes (things): {results['n_classes_things']}")
    logger.info(f"Active classes (stuff): {results['n_classes_stuff']}")
    logger.info("-" * 70)
    logger.info(f"  PQ  = {results['PQ']:.2f}   SQ  = {results['SQ']:.2f}   RQ  = {results['RQ']:.2f}")
    logger.info(f"  PQ_th = {results['PQ_th']:.2f}   SQ_th = {results['SQ_th']:.2f}   RQ_th = {results['RQ_th']:.2f}")
    logger.info(f"  PQ_st = {results['PQ_st']:.2f}   SQ_st = {results['SQ_st']:.2f}   RQ_st = {results['RQ_st']:.2f}")
    logger.info("=" * 70)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    # 1. Summary JSON
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {k: v for k, v in results.items() if k != "per_class"}
    summary["skipped_images"] = skipped_images
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # 2. Per-class CSV
    if per_class_results:
        per_class_csv_path = os.path.join(output_dir, "per_class_metrics.csv")
        with open(per_class_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category_id", "PQ", "SQ", "RQ"])
            for cat_id in sorted(per_class_results.keys()):
                m = per_class_results[cat_id]
                writer.writerow([
                    cat_id, f"{m['PQ']:.2f}", f"{m['SQ']:.2f}", f"{m['RQ']:.2f}",
                ])
        logger.info(f"Per-class metrics saved to {per_class_csv_path}")

    # 3. Optionally keep prediction PNGs for inspection
    if not save_predictions:
        import shutil
        shutil.rmtree(pred_png_dir, ignore_errors=True)
        shutil.rmtree(gt_png_dir, ignore_errors=True)
        logger.info("Cleaned up temp PNG directories (use --save-predictions to keep)")

    return results


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-View Mask2Former on ScanNet++ validation set"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/scannetpp/panoptic-segmentation/ma40.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). Overrides MODEL.WEIGHTS in config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--scannetpp-root",
        type=str,
        default=None,
        help="Override ScanNet++ data root",
    )
    parser.add_argument(
        "--panoptic-val-root",
        type=str,
        default=None,
        help="Override path to panoptic_val directory (GT annotations)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Override path to validation split file",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override NUM_CLASSES from config",
    )
    parser.add_argument(
        "--target-short-edge",
        type=int,
        default=480,
        help="Resize image shortest edge to this value for inference",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=640,
        help="Maximum image dimension after resize",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.8,
        help="Panoptic post-processing overlap threshold",
    )
    parser.add_argument(
        "--object-mask-threshold",
        type=float,
        default=0.8,
        help="Panoptic post-processing mask binarisation threshold",
    )
    parser.add_argument(
        "--max-images-per-scene",
        type=int,
        default=None,
        help="Limit images per scene (for quick evaluation)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted panoptic maps to output dir",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options via command line (e.g. MODEL.WEIGHTS /path/to/ckpt)",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # ---- Register datasets (needed for config / metadata) ----
    register_scannetpp_train_val()

    # ---- Build config ----
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_multiview_config(cfg)
    cfg.SOLVER.DPT_LR = 1e-5  # must be defined before merge
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Override checkpoint path
    if args.checkpoint:
        try:
            cfg.defrost()
        except Exception:
            pass
        cfg.MODEL.WEIGHTS = args.checkpoint

    # Backbone name (must match training)
    try:
        cfg.defrost()
    except Exception:
        pass
    cfg.MODEL.BACKBONE.NAME = "build_mapanything_multiview_backbone"
    cfg.SOLVER.AMP.ENABLED = True
    cfg.freeze()

    # ---- Resolve paths ----
    default_root = "/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/datasets/scannet/scannetpp"
    scannetpp_root = args.scannetpp_root or os.environ.get("SCANNETPP_ROOT", default_root)
    panoptic_val_root = args.panoptic_val_root or os.environ.get(
        "PANOPTIC_VAL_ROOT", f"{scannetpp_root}/panoptic_val"
    )
    val_split = args.val_split or os.environ.get(
        "VAL_SPLIT", f"{scannetpp_root}/splits/nvs_sem_val_clean.txt"
    )
    num_classes = args.num_classes or cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    checkpoint_path = cfg.MODEL.WEIGHTS

    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"ScanNet++ root: {scannetpp_root}")
    logger.info(f"Panoptic val root: {panoptic_val_root}")
    logger.info(f"Val split: {val_split}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Output dir: {args.output_dir}")

    # ---- Load model ----
    model = load_model(cfg, checkpoint_path)

    # ---- Run evaluation ----
    results = evaluate(
        model=model,
        scannetpp_root=scannetpp_root,
        panoptic_val_root=panoptic_val_root,
        val_split_file=val_split,
        output_dir=args.output_dir,
        num_classes=num_classes,
        target_short_edge=args.target_short_edge,
        max_size=args.max_size,
        overlap_threshold=args.overlap_threshold,
        object_mask_threshold=args.object_mask_threshold,
        max_images_per_scene=args.max_images_per_scene,
        save_predictions=args.save_predictions,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
