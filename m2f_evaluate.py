#!/usr/bin/env python3
"""
Multi-View Evaluation Script for Multi-View Mask2Former on ScanNet++

Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and
Recognition Quality (RQ) on the validation set using the full multi-view
pipeline with query propagation — matching training exactly.

Evaluation strategy:
1. Discover all scenes from the panoptic_val directory.
2. For each scene, select N views that have GT annotations (1 reference +
   N-1 targets), where N = cfg.MODEL.MULTIVIEW.NUM_VIEWS (default 3).
3. Feed all N views through the MapAnything backbone (cross-view attention).
4. Run reference view through pixel decoder + Mask2Former decoder with
   learnable queries.
5. Propagate refined query_embeddings to target views (query propagation,
   same as training — no warped attention mask since depth is disabled).
6. Panoptic post-process all N views independently.
7. Compare each view's prediction against its GT, compute per-view PQ.
8. Average per-view PQ within a scene, then macro-average across scenes.

This evaluates the model the way it was trained: multi-view backbone with
cross-view attention + query propagation from reference to targets.

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
import random
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
import detectron2.data.detection_utils as d2_utils
import detectron2.data.transforms as T
import detectron2.utils.comm as comm

from mask2former import add_maskformer2_config

# Official panoptic evaluation API (same as used by Detectron2 internally)
from panopticapi.utils import id2rgb, rgb2id
from panopticapi.evaluation import pq_compute

# Import model & config helpers from training script
from m2f_train_multiview import (
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
# MULTI-VIEW INFERENCE (mirrors training forward pass)
# ============================================================

@torch.no_grad()
def run_multiview_inference(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    camera_poses: List[torch.Tensor],
    camera_intrinsics: List[torch.Tensor],
    ref_view_idx: int,
    num_classes: int,
    overlap_threshold: float = 0.8,
    object_mask_threshold: float = 0.8,
) -> List[Tuple[np.ndarray, List[Dict]]]:
    """
    Run multi-view panoptic inference matching the training pipeline.

    Pipeline (mirrors MultiViewMask2Former.forward):
      1. Stack all N views into [B=1, N, 3, H, W]
      2. Backbone: all views processed together (cross-view attention)
      3. Reference view: pixel_decoder → query_propagation_decoder (learnable queries)
      4. Target views: pixel_decoder → query_propagation_decoder (propagated queries)
      5. Panoptic post-process each view independently

    Args:
        model: Trained MultiViewMask2Former
        images: List of N tensors, each [3, H, W] (0-255 range, all same H,W)
        camera_poses: List of N [4,4] camera-to-world tensors
        camera_intrinsics: List of N [3,3] intrinsic matrices
        ref_view_idx: Index of reference view (0..N-1)
        num_classes: Number of semantic classes
        overlap_threshold: Panoptic post-processing overlap threshold
        object_mask_threshold: Mask binarisation threshold

    Returns:
        List of (panoptic_map, segments_info) tuples, one per view.
        panoptic_map is [H, W] int32; segments_info is list of dicts.
    """
    device = next(model.parameters()).device
    N = len(images)
    C, H, W = images[0].shape

    # Stack into batch: [1, N, 3, H, W], [1, N, 4, 4], [1, N, 3, 3]
    images_batch = torch.stack(images, dim=0).unsqueeze(0).to(device)       # [1, N, 3, H, W]
    poses_batch = torch.stack(camera_poses, dim=0).unsqueeze(0).to(device)  # [1, N, 4, 4]
    K_batch = torch.stack(camera_intrinsics, dim=0).unsqueeze(0).to(device) # [1, N, 3, 3]

    # ----- Step 1: Backbone (all views, cross-view attention) -----
    backbone_out = model.backbone(
        images=images_batch,
        camera_poses=poses_batch,
        camera_intrinsics=K_batch,
        return_all_views=True,
    )
    all_view_features = backbone_out["all_view_features"]
    # all_view_features: {res2: [N, B, C, H', W'], ...}

    # ----- Step 2: Reference view — learnable queries -----
    ref_features = model._prepare_features_for_view(all_view_features, ref_view_idx)
    ref_mask_features, ref_multi_scale_features = model._run_pixel_decoder(ref_features)

    ref_outputs = model.query_propagation_decoder(
        ref_multi_scale_features,
        ref_mask_features,
        mask=None,
        initial_query_feat=None,   # learnable queries (same as training)
        initial_attn_mask=None,    # no warped mask (same as training)
    )

    # Extract refined queries for propagation
    ref_queries = ref_outputs["query_embeddings"]  # [Q, B, D]

    # ----- Step 3: Decode each view -----
    per_view_outputs = [None] * N

    # Reference view outputs already computed
    per_view_outputs[ref_view_idx] = ref_outputs

    # Target views: propagate reference queries
    for v_idx in range(N):
        if v_idx == ref_view_idx:
            continue
        tgt_features = model._prepare_features_for_view(all_view_features, v_idx)
        tgt_mask_features, tgt_multi_scale_features = model._run_pixel_decoder(tgt_features)

        tgt_outputs = model.query_propagation_decoder(
            tgt_multi_scale_features,
            tgt_mask_features,
            mask=None,
            initial_query_feat=ref_queries,  # propagated queries (same as training)
            initial_attn_mask=None,          # no warped mask (same as training)
        )
        per_view_outputs[v_idx] = tgt_outputs

    # ----- Step 4: Post-process each view -----
    results = []
    raw_outputs = []  # Store raw per-view outputs for cross-view metrics
    for v_idx in range(N):
        outputs = per_view_outputs[v_idx]
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]    # [B, Q, H', W']
        query_embeddings = outputs.get("query_embeddings", None)  # [Q, B, D]

        # Upsample masks to input resolution
        pred_masks_up = F.interpolate(
            pred_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        panoptic_map, segments_info = panoptic_postprocessing(
            pred_logits[0],
            pred_masks_up[0],
            num_classes=num_classes,
            overlap_threshold=overlap_threshold,
            object_mask_threshold=object_mask_threshold,
        )
        results.append((panoptic_map, segments_info))

        # Store raw outputs for cross-view metric computation
        raw_outputs.append({
            "pred_logits": pred_logits[0].cpu(),       # [Q, C+1]
            "pred_masks": pred_masks_up[0].cpu(),      # [Q, H, W]
            "query_embeddings": query_embeddings[:, 0, :].cpu() if query_embeddings is not None else None,  # [Q, D]
        })

    return results, raw_outputs

def load_model(cfg, checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = MultiViewMask2Former(cfg)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    return model


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
# CROSS-VIEW CONSISTENCY METRICS
# ============================================================

def compute_per_view_pq(
    per_view_results: List[Tuple[np.ndarray, List[Dict]]],
    view_gt_maps: List[np.ndarray],
    view_gt_segments: List[List[Dict]],
    selected_stems: List[str],
    output_dir: str,
    scene_id: str,
) -> Dict[str, Any]:
    """
    Compute PQ for each view independently and return per-view breakdown.

    Returns:
        Dict with per_view_pq (list), avg_pq, pq_variance, pq_std
    """
    num_views = len(per_view_results)
    per_view_pq = []

    for v_idx in range(num_views):
        stem = selected_stems[v_idx]
        pred_panoptic, pred_segments = per_view_results[v_idx]
        gt_panoptic_map = view_gt_maps[v_idx]
        gt_segments = view_gt_segments[v_idx]

        # Resize prediction to match GT resolution
        gt_h, gt_w = gt_panoptic_map.shape
        if pred_panoptic.shape != gt_panoptic_map.shape:
            pred_panoptic = cv2.resize(
                pred_panoptic.astype(np.float32),
                (gt_w, gt_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        # Build per-view temporary files for panopticapi evaluation
        view_out = os.path.join(output_dir, "per_view_tmp", scene_id, f"view_{v_idx}")
        os.makedirs(os.path.join(view_out, "gt"), exist_ok=True)
        os.makedirs(os.path.join(view_out, "pred"), exist_ok=True)

        file_name = f"{stem}.png"

        # Collect categories
        cat_ids = set()
        isthing_map = {}
        for seg in gt_segments:
            cid = int(seg["category_id"])
            cat_ids.add(cid)
            isthing_map[cid] = int(seg.get("isthing", 0))
        for seg in pred_segments:
            cat_ids.add(int(seg["category_id"]))

        categories = []
        for cid in sorted(cat_ids):
            categories.append({
                "id": cid,
                "name": str(cid),
                "isthing": isthing_map.get(cid, 0),
            })

        if not categories:
            per_view_pq.append(0.0)
            continue

        # Save GT PNG
        gt_png_out = re_encode_panoptic_png(gt_panoptic_map)
        gt_png_out.save(os.path.join(view_out, "gt", file_name))

        gt_ann_segments = []
        for seg in gt_segments:
            gt_ann_segments.append({
                "id": int(seg["id"]),
                "category_id": int(seg["category_id"]),
                "iscrowd": int(seg.get("iscrowd", 0)),
                "area": int((gt_panoptic_map == seg["id"]).sum()),
            })

        # Save Pred PNG
        pred_png_out = re_encode_panoptic_png(pred_panoptic)
        pred_png_out.save(os.path.join(view_out, "pred", file_name))

        pred_ann_segments = []
        for seg in pred_segments:
            pred_ann_segments.append({
                "id": int(seg["id"]),
                "category_id": int(seg["category_id"]),
                "area": int((pred_panoptic == seg["id"]).sum()),
            })

        images_meta = [{"id": 1, "file_name": stem}]

        gt_json_data = {
            "images": images_meta,
            "annotations": [{
                "image_id": 1,
                "file_name": file_name,
                "segments_info": gt_ann_segments,
            }],
            "categories": categories,
        }
        gt_json_path = os.path.join(view_out, "gt_panoptic.json")
        with open(gt_json_path, "w") as f:
            json.dump(gt_json_data, f)

        pred_json_data = {
            "images": images_meta,
            "annotations": [{
                "image_id": 1,
                "file_name": file_name,
                "segments_info": pred_ann_segments,
            }],
            "categories": categories,
        }
        pred_json_path = os.path.join(view_out, "pred_panoptic.json")
        with open(pred_json_path, "w") as f:
            json.dump(pred_json_data, f)

        try:
            pq_res = pq_compute(
                gt_json_path,
                pred_json_path,
                gt_folder=os.path.join(view_out, "gt"),
                pred_folder=os.path.join(view_out, "pred"),
            )
            all_res = pq_res.get("All", {})
            n = all_res.get("n", 0)
            pq_val = 100.0 * all_res.get("pq", 0.0) if n > 0 else 0.0
        except Exception:
            pq_val = 0.0

        per_view_pq.append(pq_val)

        # Clean up temporary files
        import shutil
        shutil.rmtree(view_out, ignore_errors=True)

    avg_pq = float(np.mean(per_view_pq)) if per_view_pq else 0.0
    pq_var = float(np.var(per_view_pq)) if len(per_view_pq) > 1 else 0.0
    pq_std = float(np.std(per_view_pq)) if len(per_view_pq) > 1 else 0.0

    return {
        "per_view_pq": per_view_pq,
        "avg_pq": avg_pq,
        "pq_variance": pq_var,
        "pq_std": pq_std,
    }


def compute_cross_view_metrics(
    raw_outputs: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute cross-view consistency metrics using raw per-view outputs.

    All metrics compare the SAME query index across different views.
    Since query propagation re-uses reference queries for target views,
    Query #k should represent the same entity in all views.

    Metrics:
        1. Class Consistency: Fraction of queries that predict the same
           class in all views (higher = more consistent).
        2. Mask IoU: Average IoU of the same query's binarised mask
           across all view pairs (higher = more spatial overlap).
        3. Query Feature Similarity: Average cosine similarity of
           query embedding vectors across all view pairs.

    Args:
        raw_outputs: List of per-view dicts, each with:
            - pred_logits: [Q, C+1] class logits
            - pred_masks:  [Q, H, W] mask logits (pre-sigmoid)
            - query_embeddings: [Q, D] query feature vectors (or None)
        num_classes: Number of valid classes (excluding no-object)

    Returns:
        Dict with class_consistency, mask_iou, query_cosine_similarity
    """
    N = len(raw_outputs)
    if N < 2:
        return {
            "class_consistency": 1.0,
            "mask_iou": 1.0,
            "query_cosine_similarity": 1.0,
        }

    Q = raw_outputs[0]["pred_logits"].shape[0]

    # ------------------------------------------------------------------
    # 1. Per-view predicted classes: [N, Q]
    # ------------------------------------------------------------------
    per_view_classes = []
    for out in raw_outputs:
        scores = F.softmax(out["pred_logits"], dim=-1)  # [Q, C+1]
        scores = scores[:, :num_classes]                 # [Q, C]
        _, labels = scores.max(dim=-1)                   # [Q]
        per_view_classes.append(labels)
    classes_stack = torch.stack(per_view_classes, dim=0)  # [N, Q]

    # Class consistency: fraction of queries where ALL views agree
    # For each query, check if the mode class == all predictions
    ref_classes = classes_stack[0]  # reference view classes
    agreement_mask = torch.ones(Q, dtype=torch.bool)
    for v in range(1, N):
        agreement_mask &= (classes_stack[v] == ref_classes)
    class_consistency = float(agreement_mask.float().mean().item())

    # ------------------------------------------------------------------
    # 2. Mask IoU across view pairs
    # ------------------------------------------------------------------
    # Binarise masks at 0.5 threshold (same as post-processing)
    per_view_binary_masks = []
    for out in raw_outputs:
        binary = (out["pred_masks"].sigmoid() > 0.5)  # [Q, H, W]
        per_view_binary_masks.append(binary)

    pair_ious = []
    for i in range(N):
        for j in range(i + 1, N):
            mask_i = per_view_binary_masks[i].float()  # [Q, H, W]
            mask_j = per_view_binary_masks[j].float()  # [Q, H, W]

            intersection = (mask_i * mask_j).sum(dim=(1, 2))  # [Q]
            union = ((mask_i + mask_j) > 0).float().sum(dim=(1, 2))  # [Q]

            # Avoid division by zero (empty masks)
            valid = union > 0
            if valid.sum() > 0:
                iou = torch.zeros(Q)
                iou[valid] = intersection[valid] / union[valid]
                pair_ious.append(iou.mean().item())
            else:
                pair_ious.append(0.0)

    mask_iou = float(np.mean(pair_ious)) if pair_ious else 0.0

    # ------------------------------------------------------------------
    # 3. Query feature cosine similarity across view pairs
    # ------------------------------------------------------------------
    has_embeddings = all(
        out.get("query_embeddings") is not None for out in raw_outputs
    )

    if has_embeddings:
        per_view_embeds = []
        for out in raw_outputs:
            emb = out["query_embeddings"]  # [Q, D]
            # L2 normalise for cosine similarity
            emb_norm = F.normalize(emb, p=2, dim=-1)
            per_view_embeds.append(emb_norm)

        pair_cosines = []
        for i in range(N):
            for j in range(i + 1, N):
                # Cosine similarity per query: dot product of normalised vectors
                cos_sim = (per_view_embeds[i] * per_view_embeds[j]).sum(dim=-1)  # [Q]
                pair_cosines.append(cos_sim.mean().item())

        query_cosine_similarity = float(np.mean(pair_cosines)) if pair_cosines else 0.0
    else:
        query_cosine_similarity = float("nan")

    return {
        "class_consistency": class_consistency,
        "mask_iou": mask_iou,
        "query_cosine_similarity": query_cosine_similarity,
    }


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

    Uses the same pipeline as training:
      - detectron2 read_image (handles EXIF rotation, returns HWC uint8)
      - detectron2 ResizeShortestEdge (same rounding as training)
      - Pad to be divisible by 32 (value=128, same as training)
      - torch.as_tensor → float  (training uses int, but backbone casts)

    Returns:
        image: [3, H, W] float32 tensor (0-255 range, RGB)
        original_size: (orig_H, orig_W)
    """
    # Read image (same as training: utils.read_image with RGB format)
    img = d2_utils.read_image(image_path, format="RGB")  # HWC uint8 np
    orig_h, orig_w = img.shape[:2]

    # Resize using detectron2 transform (same as training dataset mapper)
    resize_tfm = T.ResizeShortestEdge(target_short_edge, max_size)
    transform = resize_tfm.get_transform(img)
    img = transform.apply_image(img)

    new_h, new_w = img.shape[:2]

    # Pad to be divisible by 32 (same as training, value=128)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                     mode="constant", constant_values=128)

    # To tensor [3, H, W] float32 (training uses torch.as_tensor which
    # preserves uint8, but backbone internally casts to float; using
    # .float() here is equivalent and matches m2f_inference.py)
    img_tensor = torch.from_numpy(
        np.ascontiguousarray(img.transpose(2, 0, 1))
    ).float()

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
    num_views: int = 3,
    target_short_edge: int = 480,
    max_size: int = 640,
    overlap_threshold: float = 0.8,
    object_mask_threshold: float = 0.8,
    max_images_per_scene: Optional[int] = None,
    save_predictions: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run multi-view per-scene evaluation over all scenes in panoptic_val.

    For each scene:
      1. Select N views that have GT annotations (1 ref + N-1 targets).
      2. Feed all N views through the backbone (cross-view attention).
      3. Run reference view with learnable queries, propagate to targets.
      4. Panoptic post-process all N views.
      5. Compute PQ for each view against its GT, average within scene.

    After all scenes, macro-average the per-scene metrics.

    This matches the training forward pass: multi-view backbone →
    query propagation from reference to targets.

    Args:
        model: Trained MultiViewMask2Former (already on GPU)
        scannetpp_root: Path to ScanNet++ data root
        panoptic_val_root: Path to panoptic_val directory with GT JSONs + PNGs
        val_split_file: Unused (scenes discovered from panoptic_val directory)
        output_dir: Directory for output CSVs / predictions
        num_classes: Number of semantic classes
        num_views: Number of views per scene (1 ref + N-1 targets)
        target_short_edge: Resize short edge for inference
        max_size: Maximum image dimension
        overlap_threshold: Panoptic post-processing overlap threshold
        object_mask_threshold: Mask binarisation threshold
        max_images_per_scene: Unused (kept for API compatibility)
        save_predictions: Save predicted panoptic PNGs to output dir
        seed: Random seed for view selection (None = random)

    Returns:
        Dict with averaged PQ, SQ, RQ metrics and per-scene breakdown
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Discover all scenes from panoptic_val directory
    # ---------------------------------------------------------------
    panoptic_val_path = Path(panoptic_val_root)
    val_scenes = sorted([
        d.name for d in panoptic_val_path.iterdir()
        if d.is_dir()
    ])

    logger.info(f"Discovered {len(val_scenes)} validation scenes in {panoptic_val_root}")
    logger.info(f"Multi-view evaluation: {num_views} views per scene (1 ref + {num_views - 1} targets)")
    logger.info(f"Image short edge: {target_short_edge}, max size: {max_size}")

    model.eval()
    device = next(model.parameters()).device

    # Per-scene results accumulator
    per_scene_results = {}
    scenes_evaluated = 0
    scenes_skipped = 0

    # RNG for reproducible view selection
    rng = np.random.default_rng(seed)

    for scene_idx, scene_id in enumerate(tqdm(val_scenes, desc="Scenes")):
        scene_panoptic_dir = panoptic_val_path / scene_id

        if not scene_panoptic_dir.exists():
            logger.warning(f"No panoptic GT for scene {scene_id}, skipping")
            scenes_skipped += 1
            continue

        # ---------------------------------------------------------------
        # 1. Find all GT views with both JSON and PNG
        # ---------------------------------------------------------------
        gt_json_files = sorted(scene_panoptic_dir.glob("*.json"))
        if not gt_json_files:
            logger.warning(f"No GT annotations for scene {scene_id}, skipping")
            scenes_skipped += 1
            continue

        # Filter to views that have both JSON and PNG
        available_stems = []
        for jf in gt_json_files:
            if (scene_panoptic_dir / f"{jf.stem}.png").exists():
                available_stems.append(jf.stem)

        if len(available_stems) < num_views:
            logger.warning(
                f"Scene {scene_id}: only {len(available_stems)} GT views, "
                f"need {num_views}, skipping"
            )
            scenes_skipped += 1
            continue

        # ---------------------------------------------------------------
        # 2. Load camera data for this scene
        # ---------------------------------------------------------------
        scene = ScanNetPPScene(scene_id, scannetpp_root)
        intrinsics = None
        frames = []

        transforms_path = scene.dslr_nerfstudio_transforms_undistorted
        if not transforms_path.exists():
            transforms_path = scene.dslr_nerfstudio_transforms

        if transforms_path.exists():
            intrinsics, frames = read_nerfstudio_transforms(str(transforms_path))
        else:
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
            else:
                logger.warning(f"No transforms for scene {scene_id}, skipping")
                scenes_skipped += 1
                continue

        if not frames or intrinsics is None:
            logger.warning(f"No frames/intrinsics for scene {scene_id}, skipping")
            scenes_skipped += 1
            continue

        K = build_intrinsic_matrix(intrinsics)
        K_tensor = torch.from_numpy(K).float()

        # Build frame lookup by stem
        frame_lookup = {Path(f["file_path"]).stem: f for f in frames}

        # Get image directory
        image_dir = scene.dslr_undistorted_dir
        if not image_dir.exists():
            image_dir = scene.dslr_resized_dir

        # ---------------------------------------------------------------
        # 3. Select N views that have GT + camera pose + image file
        # ---------------------------------------------------------------
        # Filter available_stems to those with camera data and images
        valid_stems = []
        for stem in available_stems:
            # Check camera pose exists
            fi = frame_lookup.get(stem)
            if fi is None:
                # Try partial match
                for key in frame_lookup:
                    if key.startswith(stem):
                        fi = frame_lookup[key]
                        break
            if fi is None:
                continue
            # Check image file exists
            found_img = False
            for ext in [".JPG", ".jpg", ".png"]:
                if (image_dir / f"{stem}{ext}").exists():
                    found_img = True
                    break
            if found_img:
                valid_stems.append(stem)

        if len(valid_stems) < num_views:
            logger.warning(
                f"Scene {scene_id}: only {len(valid_stems)} valid views "
                f"(with GT + pose + image), need {num_views}, skipping"
            )
            scenes_skipped += 1
            continue

        # Randomly select N views; view 0 = reference
        selected_indices = rng.choice(len(valid_stems), size=num_views, replace=False)
        selected_stems = [valid_stems[i] for i in sorted(selected_indices)]
        ref_view_idx = 0  # First selected view is reference

        # ---------------------------------------------------------------
        # 4. Load images, poses, intrinsics for selected views
        # ---------------------------------------------------------------
        view_images = []
        view_poses = []
        view_K_scaled = []
        view_gt_maps = []
        view_gt_segments = []

        try:
            # We need all images at the same spatial resolution for batching.
            # Load + resize each view identically.
            for stem in selected_stems:
                # --- Image ---
                image_path = None
                for ext in [".JPG", ".jpg", ".png"]:
                    candidate = image_dir / f"{stem}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        break

                img_tensor, orig_size = load_and_prepare_image(
                    image_path, target_short_edge, max_size
                )

                # --- Camera pose ---
                fi = frame_lookup.get(stem)
                if fi is None:
                    for key in frame_lookup:
                        if key.startswith(stem):
                            fi = frame_lookup[key]
                            break

                c2w = fi.get("camera_to_world")
                if c2w is None:
                    if "transform_matrix" in fi:
                        c2w_opengl = np.array(fi["transform_matrix"], dtype=np.float32)
                        convert_mat = np.diag([1, -1, -1, 1]).astype(np.float32)
                        c2w = c2w_opengl @ convert_mat
                if isinstance(c2w, np.ndarray):
                    c2w = torch.from_numpy(c2w).float()
                else:
                    c2w = torch.tensor(c2w).float()

                # --- Scale intrinsics ---
                orig_h, orig_w = orig_size
                _, resized_h, resized_w = img_tensor.shape
                scale_x = resized_w / orig_w
                scale_y = resized_h / orig_h
                Ks = K_tensor.clone()
                Ks[0, 0] *= scale_x
                Ks[1, 1] *= scale_y
                Ks[0, 2] *= scale_x
                Ks[1, 2] *= scale_y

                # --- GT ---
                gt_png_path = scene_panoptic_dir / f"{stem}.png"
                gt_json_path = scene_panoptic_dir / f"{stem}.json"
                gt_panoptic_map, gt_segments = load_gt_panoptic(
                    str(gt_png_path), str(gt_json_path)
                )

                view_images.append(img_tensor)
                view_poses.append(c2w)
                view_K_scaled.append(Ks)
                view_gt_maps.append(gt_panoptic_map)
                view_gt_segments.append(gt_segments)

            # All images must be the same H, W for batching.
            # They should be (same scene, same camera, same resize params),
            # but enforce by checking and padding to largest if needed.
            shapes = [img.shape for img in view_images]
            max_h = max(s[1] for s in shapes)
            max_w = max(s[2] for s in shapes)
            for i in range(len(view_images)):
                c, h, w = view_images[i].shape
                if h < max_h or w < max_w:
                    padded = torch.full((c, max_h, max_w), 128.0)
                    padded[:, :h, :w] = view_images[i]
                    view_images[i] = padded

            # ---------------------------------------------------------------
            # 5. Run multi-view inference
            # ---------------------------------------------------------------
            per_view_results, raw_outputs = run_multiview_inference(
                model=model,
                images=view_images,
                camera_poses=view_poses,
                camera_intrinsics=view_K_scaled,
                ref_view_idx=ref_view_idx,
                num_classes=num_classes,
                overlap_threshold=overlap_threshold,
                object_mask_threshold=object_mask_threshold,
            )

            # ---------------------------------------------------------------
            # 6. Evaluate each view against its GT
            # ---------------------------------------------------------------
            scene_output_dir = os.path.join(output_dir, "per_scene", scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)

            gt_png_dir = os.path.join(scene_output_dir, "gt_pngs")
            pred_png_dir = os.path.join(scene_output_dir, "pred_pngs")
            os.makedirs(gt_png_dir, exist_ok=True)
            os.makedirs(pred_png_dir, exist_ok=True)

            all_gt_annotations = []
            all_pred_annotations = []
            all_images_meta = []

            # Collect categories from all views in this scene
            scene_cat_ids = set()
            gt_isthing = {}

            for v_idx in range(num_views):
                stem = selected_stems[v_idx]
                pred_panoptic, pred_segments = per_view_results[v_idx]
                gt_panoptic_map = view_gt_maps[v_idx]
                gt_segments = view_gt_segments[v_idx]

                # Resize prediction to match GT resolution
                gt_h, gt_w = gt_panoptic_map.shape
                if pred_panoptic.shape != gt_panoptic_map.shape:
                    pred_panoptic = cv2.resize(
                        pred_panoptic.astype(np.float32),
                        (gt_w, gt_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)

                # Collect categories
                for seg in gt_segments:
                    cid = int(seg["category_id"])
                    scene_cat_ids.add(cid)
                    if cid not in gt_isthing:
                        gt_isthing[cid] = int(seg.get("isthing", 0))
                for seg in pred_segments:
                    scene_cat_ids.add(int(seg["category_id"]))

                img_id = v_idx + 1
                file_name = f"{stem}.png"

                # Save GT PNG
                gt_png_out = re_encode_panoptic_png(gt_panoptic_map)
                gt_png_out.save(os.path.join(gt_png_dir, file_name))

                gt_ann_segments = []
                for seg in gt_segments:
                    gt_ann_segments.append({
                        "id": int(seg["id"]),
                        "category_id": int(seg["category_id"]),
                        "iscrowd": int(seg.get("iscrowd", 0)),
                        "area": int((gt_panoptic_map == seg["id"]).sum()),
                    })

                # Save Pred PNG
                pred_png_out = re_encode_panoptic_png(pred_panoptic)
                pred_png_out.save(os.path.join(pred_png_dir, file_name))

                pred_ann_segments = []
                for seg in pred_segments:
                    pred_ann_segments.append({
                        "id": int(seg["id"]),
                        "category_id": int(seg["category_id"]),
                        "area": int((pred_panoptic == seg["id"]).sum()),
                    })

                all_images_meta.append({"id": img_id, "file_name": stem})
                all_gt_annotations.append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "segments_info": gt_ann_segments,
                })
                all_pred_annotations.append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "segments_info": pred_ann_segments,
                })

            # Build scene-specific categories
            scene_categories = []
            for cid in sorted(scene_cat_ids):
                scene_categories.append({
                    "id": cid,
                    "name": str(cid),
                    "isthing": gt_isthing.get(cid, 0),
                })

            if not scene_categories:
                logger.warning(f"No categories in GT/pred for scene {scene_id}, skipping")
                scenes_skipped += 1
                continue

            # Build COCO-format JSONs for all views in this scene
            gt_json_data = {
                "images": all_images_meta,
                "annotations": all_gt_annotations,
                "categories": scene_categories,
            }
            gt_json_out = os.path.join(scene_output_dir, "gt_panoptic.json")
            with open(gt_json_out, "w") as f:
                json.dump(gt_json_data, f)

            pred_json_data = {
                "images": all_images_meta,
                "annotations": all_pred_annotations,
                "categories": scene_categories,
            }
            pred_json_out = os.path.join(scene_output_dir, "pred_panoptic.json")
            with open(pred_json_out, "w") as f:
                json.dump(pred_json_data, f)

            # Run panopticapi evaluation for this scene (across all N views)
            pq_res = pq_compute(
                gt_json_out,
                pred_json_out,
                gt_folder=gt_png_dir,
                pred_folder=pred_png_dir,
            )

            # Extract metrics, handling zero-category cases gracefully
            def _safe_get(d, key):
                val = d.get(key, {})
                n = val.get("n", 0)
                if n == 0:
                    return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}
                return val

            all_res = _safe_get(pq_res, "All")
            things_res = _safe_get(pq_res, "Things")
            stuff_res = _safe_get(pq_res, "Stuff")

            # ---------------------------------------------------------------
            # 6b. Compute per-view PQ breakdown
            # ---------------------------------------------------------------
            per_view_pq_metrics = compute_per_view_pq(
                per_view_results=per_view_results,
                view_gt_maps=view_gt_maps,
                view_gt_segments=view_gt_segments,
                selected_stems=selected_stems,
                output_dir=output_dir,
                scene_id=scene_id,
            )

            # ---------------------------------------------------------------
            # 6c. Compute cross-view consistency metrics
            # ---------------------------------------------------------------
            cross_view_metrics = compute_cross_view_metrics(
                raw_outputs=raw_outputs,
                num_classes=num_classes,
            )

            scene_metrics = {
                "PQ": 100 * all_res["pq"],
                "SQ": 100 * all_res["sq"],
                "RQ": 100 * all_res["rq"],
                "PQ_th": 100 * things_res["pq"],
                "SQ_th": 100 * things_res["sq"],
                "RQ_th": 100 * things_res["rq"],
                "PQ_st": 100 * stuff_res["pq"],
                "SQ_st": 100 * stuff_res["sq"],
                "RQ_st": 100 * stuff_res["rq"],
                "n_classes_all": all_res["n"],
                "n_classes_things": things_res["n"],
                "n_classes_stuff": stuff_res["n"],
                "num_views": num_views,
                "reference_view": selected_stems[ref_view_idx],
                "target_views": [s for i, s in enumerate(selected_stems) if i != ref_view_idx],
                # Per-view PQ breakdown
                "per_view_pq": per_view_pq_metrics["per_view_pq"],
                "avg_per_view_pq": per_view_pq_metrics["avg_pq"],
                "pq_variance": per_view_pq_metrics["pq_variance"],
                "pq_std": per_view_pq_metrics["pq_std"],
                # Cross-view consistency metrics
                "class_consistency": cross_view_metrics["class_consistency"],
                "mask_iou": cross_view_metrics["mask_iou"],
                "query_cosine_similarity": cross_view_metrics["query_cosine_similarity"],
            }

            per_scene_results[scene_id] = scene_metrics
            scenes_evaluated += 1

            logger.info(
                f"  Scene {scene_id} ({num_views}v, ref={selected_stems[ref_view_idx]}): "
                f"PQ={scene_metrics['PQ']:.2f}  "
                f"SQ={scene_metrics['SQ']:.2f}  "
                f"RQ={scene_metrics['RQ']:.2f}  |"
                f"  Per-View PQ: {[f'{p:.1f}' for p in scene_metrics['per_view_pq']]}  "
                f"Var={scene_metrics['pq_variance']:.2f}  |"
                f"  ClassCons={scene_metrics['class_consistency']:.3f}  "
                f"MaskIoU={scene_metrics['mask_iou']:.3f}  "
                f"QuerCos={scene_metrics['query_cosine_similarity']:.3f}"
            )

            # Clean up per-scene temp PNGs unless user wants them
            if not save_predictions:
                import shutil
                shutil.rmtree(gt_png_dir, ignore_errors=True)
                shutil.rmtree(pred_png_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"Error processing scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            scenes_skipped += 1
            continue

    # ---------------------------------------------------------------
    # Average metrics across all scenes
    # ---------------------------------------------------------------
    logger.info(f"\nEvaluation complete: {scenes_evaluated} scenes evaluated, "
                f"{scenes_skipped} scenes skipped")

    if scenes_evaluated == 0:
        logger.error("No scenes were successfully evaluated!")
        return {}

    # Compute macro-average across scenes
    metric_keys = ["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]
    avg_results = OrderedDict()
    for key in metric_keys:
        values = [per_scene_results[s][key] for s in per_scene_results]
        avg_results[key] = np.mean(values)

    # Average per-view PQ metrics
    avg_per_view_pq_values = [per_scene_results[s]["avg_per_view_pq"] for s in per_scene_results]
    avg_results["avg_per_view_pq"] = float(np.mean(avg_per_view_pq_values))
    pq_var_values = [per_scene_results[s]["pq_variance"] for s in per_scene_results]
    avg_results["pq_variance"] = float(np.mean(pq_var_values))
    pq_std_values = [per_scene_results[s]["pq_std"] for s in per_scene_results]
    avg_results["pq_std"] = float(np.mean(pq_std_values))

    # Compute per-view-index averages across scenes (View 0, View 1, ...)
    max_views = max(len(per_scene_results[s]["per_view_pq"]) for s in per_scene_results)
    for vi in range(max_views):
        vi_pqs = [per_scene_results[s]["per_view_pq"][vi]
                  for s in per_scene_results
                  if vi < len(per_scene_results[s]["per_view_pq"])]
        avg_results[f"PQ_view_{vi}"] = float(np.mean(vi_pqs)) if vi_pqs else 0.0

    # Average cross-view consistency metrics
    cc_values = [per_scene_results[s]["class_consistency"] for s in per_scene_results]
    avg_results["class_consistency"] = float(np.mean(cc_values))
    miou_values = [per_scene_results[s]["mask_iou"] for s in per_scene_results]
    avg_results["mask_iou"] = float(np.mean(miou_values))
    qcos_values = [per_scene_results[s]["query_cosine_similarity"] for s in per_scene_results
                   if not np.isnan(per_scene_results[s]["query_cosine_similarity"])]
    avg_results["query_cosine_similarity"] = float(np.mean(qcos_values)) if qcos_values else float("nan")

    avg_results["num_views_per_scene"] = num_views
    avg_results["num_scenes_evaluated"] = scenes_evaluated
    avg_results["num_scenes_skipped"] = scenes_skipped
    avg_results["num_scenes_total"] = len(val_scenes)

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"EVALUATION RESULTS — Multi-View ({num_views} views/scene, query propagation)")
    logger.info("=" * 80)
    logger.info(f"Scenes evaluated: {scenes_evaluated} / {len(val_scenes)}")
    logger.info(f"Scenes skipped:   {scenes_skipped}")
    logger.info(f"Views per scene:  {num_views} (1 ref + {num_views - 1} targets)")
    logger.info(f"Image resolution: {target_short_edge}×{max_size} (short edge × max size)")
    logger.info(f"Post-processing:  overlap_thr={overlap_threshold}, mask_thr={object_mask_threshold}")
    logger.info("  Note: Post-processing thresholds affect PQ/SQ/RQ, NOT cross-view metrics.")
    logger.info("-" * 80)
    logger.info("  [Panoptic Quality]")
    logger.info(f"  PQ    = {avg_results['PQ']:.2f}   SQ    = {avg_results['SQ']:.2f}   RQ    = {avg_results['RQ']:.2f}")
    logger.info(f"  PQ_th = {avg_results['PQ_th']:.2f}   SQ_th = {avg_results['SQ_th']:.2f}   RQ_th = {avg_results['RQ_th']:.2f}")
    logger.info(f"  PQ_st = {avg_results['PQ_st']:.2f}   SQ_st = {avg_results['SQ_st']:.2f}   RQ_st = {avg_results['RQ_st']:.2f}")
    logger.info("-" * 80)
    logger.info("  [Per-View PQ Breakdown]")
    view_pq_parts = [f"  View {vi} ({'Ref' if vi == 0 else 'Tgt'}) = {avg_results.get(f'PQ_view_{vi}', 0.0):.2f}"
                     for vi in range(max_views)]
    logger.info("  " + "   ".join(view_pq_parts))
    logger.info(f"  Average PQ = {avg_results['avg_per_view_pq']:.2f}   "
                f"Variance = {avg_results['pq_variance']:.4f}   "
                f"Std = {avg_results['pq_std']:.2f}")
    logger.info("-" * 80)
    logger.info("  [Cross-View Consistency - Query Propagation Performance]")
    logger.info(f"  Class Consistency        = {avg_results['class_consistency']:.4f}  (target: >0.80, measures semantic identity)")
    qcos_str = f"{avg_results['query_cosine_similarity']:.4f}" if not np.isnan(avg_results['query_cosine_similarity']) else "N/A"
    logger.info(f"  Query Cosine Similarity  = {qcos_str}  (target: >0.85, measures feature alignment)")
    logger.info(f"  Mask IoU                 = {avg_results['mask_iou']:.4f}  (informational only - no geometric coupling)")
    logger.info("  ")
    logger.info("  Note: Mask IoU is expected to be lower without warped attention guidance.")
    logger.info("        Focus on Class Consistency + Query Similarity for query propagation quality.")
    logger.info("=" * 80)

    # Per-scene breakdown table
    logger.info("\nPer-Scene Breakdown:")
    logger.info(f"{'Scene':<25} {'Ref View':<15} {'PQ':>6} {'SQ':>6} {'RQ':>6}  "
                f"{'AvgPQ':>6} {'PQVar':>7} {'ClsCon':>7} {'MskIoU':>7} {'QCos':>6}")
    logger.info("-" * 110)
    for scene_id in sorted(per_scene_results.keys()):
        m = per_scene_results[scene_id]
        qcos_val = f"{m['query_cosine_similarity']:.3f}" if not np.isnan(m['query_cosine_similarity']) else "  N/A"
        logger.info(
            f"{scene_id:<25} {m['reference_view']:<15} "
            f"{m['PQ']:6.2f} {m['SQ']:6.2f} {m['RQ']:6.2f}  "
            f"{m['avg_per_view_pq']:6.2f} {m['pq_variance']:7.4f} "
            f"{m['class_consistency']:7.4f} {m['mask_iou']:7.4f} {qcos_val:>6}"
        )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    # 1. Summary JSON (averages)
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in avg_results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # 2. Per-scene CSV
    per_scene_csv_path = os.path.join(output_dir, "per_scene_metrics.csv")
    # Build dynamic per-view PQ column headers
    view_pq_headers = [f"PQ_view_{vi}" for vi in range(max_views)]
    with open(per_scene_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scene_id", "reference_view", "num_views", "PQ", "SQ", "RQ",
             "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]
            + view_pq_headers
            + ["avg_per_view_pq", "pq_variance", "pq_std",
               "class_consistency", "mask_iou", "query_cosine_similarity"]
        )
        for scene_id in sorted(per_scene_results.keys()):
            m = per_scene_results[scene_id]
            view_pq_vals = [f"{m['per_view_pq'][vi]:.2f}"
                           if vi < len(m['per_view_pq']) else ""
                           for vi in range(max_views)]
            qcos_val = f"{m['query_cosine_similarity']:.4f}" if not np.isnan(m['query_cosine_similarity']) else ""
            writer.writerow(
                [scene_id, m["reference_view"], m["num_views"],
                 f"{m['PQ']:.2f}", f"{m['SQ']:.2f}", f"{m['RQ']:.2f}",
                 f"{m['PQ_th']:.2f}", f"{m['SQ_th']:.2f}", f"{m['RQ_th']:.2f}",
                 f"{m['PQ_st']:.2f}", f"{m['SQ_st']:.2f}", f"{m['RQ_st']:.2f}"]
                + view_pq_vals
                + [f"{m['avg_per_view_pq']:.2f}", f"{m['pq_variance']:.4f}", f"{m['pq_std']:.2f}",
                   f"{m['class_consistency']:.4f}", f"{m['mask_iou']:.4f}", qcos_val]
            )
    logger.info(f"Per-scene metrics saved to {per_scene_csv_path}")

    # 3. Clean up per_scene directory if not saving predictions
    if not save_predictions:
        import shutil
        per_scene_dir = os.path.join(output_dir, "per_scene")
        shutil.rmtree(per_scene_dir, ignore_errors=True)
        logger.info("Cleaned up per-scene temp directories (use --save-predictions to keep)")

    return {**avg_results, "per_scene": per_scene_results}


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-View Mask2Former on ScanNet++ validation set (per-scene)"
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
        help="Path to validation split file (unused — scenes are discovered from panoptic-val-root)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reference view selection (None = random each run)",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="Number of views per scene (1 ref + N-1 targets). Defaults to config NUM_VIEWS.",
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
    num_views = args.num_views or cfg.MODEL.MULTIVIEW.NUM_VIEWS
    checkpoint_path = cfg.MODEL.WEIGHTS

    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"ScanNet++ root: {scannetpp_root}")
    logger.info(f"Panoptic val root: {panoptic_val_root}")
    logger.info(f"Val split: {val_split}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Num views: {num_views}")
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
        num_views=num_views,
        target_short_edge=args.target_short_edge,
        max_size=args.max_size,
        overlap_threshold=args.overlap_threshold,
        object_mask_threshold=args.object_mask_threshold,
        max_images_per_scene=args.max_images_per_scene,
        save_predictions=args.save_predictions,
        seed=args.seed,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
