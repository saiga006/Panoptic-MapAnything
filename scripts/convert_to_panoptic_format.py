#!/usr/bin/env python3
"""
Convert ScanNet++ semantics_2d.py output to Detectron2 panoptic format.

Input format (from semantics_2d.py):
- obj_ids/<scene_id>/<image_name>.pth     (instance IDs as PyTorch tensor)
- semantics/<scene_id>/<image_name>.png   (semantic class IDs as PNG)

Output format (for Detectron2/Mask2Former):
- panoptic/<scene_id>/<image_stem>.png    (panoptic IDs: semantic_id * 1000 + instance_id)
- panoptic/<scene_id>/<image_stem>.json   (segments_info with category_id, id, area, etc.)

Panoptic ID encoding:
  panoptic_id = semantic_id * LABEL_DIVISOR + instance_id
  
  Where LABEL_DIVISOR = 1000 (standard Detectron2 encoding)

Usage:
  python convert_to_panoptic_format.py \
    --input_dir /path/to/panoptic_annotations \
    --output_dir /path/to/panoptic \
    --scene_list /path/to/nvs_sem_train_subset150.txt \
    --semantic_classes /path/to/semantic_classes.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import cv2
import torch


# Panoptic ID encoding for ScanNet++ (2878 semantic classes)
# LABEL_DIVISOR must be > max instances per class
# ScanNet++ can have 100s of instances per scene, use 10000 to be safe
LABEL_DIVISOR = 10000
IGNORE_LABEL = 255


def load_semantic_classes(semantic_classes_path: str, instance_classes_path: str = None) -> dict:
    """Load semantic class names and create category info."""
    with open(semantic_classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # Load instance classes if provided
    instance_class_set = None
    if instance_classes_path and os.path.exists(instance_classes_path):
        with open(instance_classes_path, 'r') as f:
            instance_class_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(instance_class_set)} instance classes from file")

    # Create category info dict
    # For ScanNet++, we need to define which are "things" vs "stuff"
    # If instance_classes file is provided, use it.
    # Otherwise, use a hardcoded list of common stuff classes.
    
    stuff_classes = {
        'wall', 'floor', 'ceiling', 'door', 'window', 'curtain',
        'blinds', 'shower curtain', 'rug', 'carpet', 'mat',
        'tile', 'column', 'pipes', 'stairs'
    }
    
    categories = {}
    for idx, class_name in enumerate(classes):
        if instance_class_set is not None:
             is_thing = class_name in instance_class_set
        else:
            is_thing = class_name.lower() not in stuff_classes
            
        categories[idx] = {
            'id': idx,
            'name': class_name,
            'isthing': is_thing,
        }
    
    return categories


def convert_to_panoptic_id(semantic_id: np.ndarray, instance_id: np.ndarray) -> np.ndarray:
    """
    Convert semantic + instance IDs to panoptic ID.
    
    panoptic_id = semantic_id * LABEL_DIVISOR + instance_id
    
    For stuff classes (no instances): instance_id = 0
    For thing classes: instance_id > 0
    """
    # Handle ignore regions
    valid_mask = (semantic_id != IGNORE_LABEL) & (instance_id >= 0)
    
    panoptic_id = np.zeros_like(semantic_id, dtype=np.int32)
    panoptic_id[valid_mask] = (
        semantic_id[valid_mask].astype(np.int32) * LABEL_DIVISOR + 
        np.clip(instance_id[valid_mask], 0, LABEL_DIVISOR - 1).astype(np.int32)
    )
    
    # Set ignore regions to 0 (background)
    panoptic_id[~valid_mask] = 0
    
    return panoptic_id


def create_segments_info(
    semantic_map: np.ndarray, 
    instance_map: np.ndarray,
    panoptic_map: np.ndarray,
    categories: dict,
) -> list:
    """
    Create segments_info list for the panoptic annotation.
    
    Each segment has:
    - id: unique panoptic ID
    - category_id: semantic class ID  
    - area: number of pixels
    - iscrowd: 0
    - isthing: whether it's a thing class
    """
    segments_info = []
    
    # Get unique panoptic IDs (excluding 0 = background)
    unique_ids = np.unique(panoptic_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    for pan_id in unique_ids:
        mask = panoptic_map == pan_id
        area = int(mask.sum())
        
        if area < 10:  # Skip tiny segments
            continue
        
        # Decode semantic ID
        semantic_id = int(pan_id // LABEL_DIVISOR)
        instance_id = int(pan_id % LABEL_DIVISOR)
        
        # Get category info
        if semantic_id in categories:
            cat_info = categories[semantic_id]
            isthing = cat_info.get('isthing', True)
        else:
            isthing = True
        
        segment = {
            'id': int(pan_id),
            'category_id': semantic_id,
            'area': area,
            'iscrowd': 0,
            'isthing': int(isthing),
        }
        segments_info.append(segment)
    
    return segments_info


def convert_scene(
    scene_id: str,
    input_dir: Path,
    output_dir: Path,
    categories: dict,
) -> int:
    """Convert one scene from semantics_2d format to panoptic format."""
    
    obj_ids_dir = input_dir / 'obj_ids' / scene_id
    semantics_dir = input_dir / 'semantics' / scene_id
    output_scene_dir = output_dir / scene_id
    
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Check what files exist
    if not obj_ids_dir.exists() or not semantics_dir.exists():
        print(f"  Skipping {scene_id}: missing obj_ids or semantics dir")
        return 0
    
    # Get list of images (from obj_ids which are .pth files)
    pth_files = sorted(obj_ids_dir.glob('*.pth'))
    
    converted = 0
    for pth_path in pth_files:
        # Get corresponding semantic PNG
        image_name = pth_path.stem  # e.g., "DSC00001.JPG"
        
        # Try different extensions
        semantic_path = None
        for ext in ['.png', '.PNG']:
            candidate = semantics_dir / f"{image_name}{ext}"
            if candidate.exists():
                semantic_path = candidate
                break
        
        if semantic_path is None:
            # Semantic file might have image extension in name: DSC00001.JPG.png
            for ext in ['.png', '.PNG']:
                candidate = semantics_dir / f"{image_name}{ext}"
                if candidate.exists():
                    semantic_path = candidate
                    break
        
        if semantic_path is None:
            continue
        
        # Load instance IDs (torch.save stores numpy arrays directly)
        instance_map = torch.load(pth_path)
        
        # Convert to numpy if it's a tensor
        if hasattr(instance_map, 'numpy'):
            instance_map = instance_map.numpy()
        elif not isinstance(instance_map, np.ndarray):
            instance_map = np.array(instance_map)
        
        # Load semantic IDs
        semantic_map = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        
        if semantic_map is None:
            print(f"  Failed to load: {semantic_path}")
            continue
        
        # Ensure same shape
        if instance_map.shape != semantic_map.shape:
            # Resize instance map to match semantic map
            instance_map = cv2.resize(
                instance_map.astype(np.float32), 
                (semantic_map.shape[1], semantic_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
        
        # Convert to panoptic format
        panoptic_map = convert_to_panoptic_id(semantic_map, instance_map)
        
        # Create segments info
        segments_info = create_segments_info(
            semantic_map, instance_map, panoptic_map, categories
        )
        
        # Save panoptic PNG (as 3-channel RGB for standard format)
        # Encode panoptic_id as: R + G*256 + B*256*256
        output_stem = image_name.replace('.JPG', '').replace('.jpg', '').replace('.png', '')
        
        # Save as single-channel for simplicity (works if IDs < 65536)
        panoptic_path = output_scene_dir / f"{output_stem}.png"
        
        # Save as 16-bit grayscale if IDs are small enough
        if panoptic_map.max() < 65536:
            cv2.imwrite(str(panoptic_path), panoptic_map.astype(np.uint16))
        else:
            # Use RGB encoding for large IDs
            # NOTE: cv2.imwrite interprets input as BGR, so the on-disk byte order
            # will have R and B swapped. The dataset mapper compensates by using
            # B + G*256 + R*65536 when decoding. Keep consistent with generate_panoptic_labels.py.
            rgb = np.zeros((*panoptic_map.shape, 3), dtype=np.uint8)
            rgb[:, :, 0] = panoptic_map % 256
            rgb[:, :, 1] = (panoptic_map // 256) % 256
            rgb[:, :, 2] = (panoptic_map // 65536) % 256
            cv2.imwrite(str(panoptic_path), rgb)
        
        # Save segments info JSON
        info_path = output_scene_dir / f"{output_stem}.json"
        with open(info_path, 'w') as f:
            json.dump(segments_info, f)
        
        converted += 1
    
    return converted


def main():
    parser = argparse.ArgumentParser(description='Convert ScanNet++ to Detectron2 panoptic format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory (panoptic_annotations from semantics_2d.py)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for Detectron2 panoptic format')
    parser.add_argument('--scene_list', type=str, required=True,
                        help='Path to scene list file')
    parser.add_argument('--semantic_classes', type=str, required=True,
                        help='Path to semantic_classes.txt')
    parser.add_argument('--instance_classes', type=str, default=None,
                        help='Path to instance_classes.txt (optional)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Load scene list
    with open(args.scene_list, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Converting {len(scene_ids)} scenes")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Load semantic classes
    categories = load_semantic_classes(args.semantic_classes, args.instance_classes)
    print(f"Loaded {len(categories)} semantic classes")
    
    # Convert each scene
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_converted = 0
    for scene_id in tqdm(scene_ids, desc='Converting scenes'):
        converted = convert_scene(scene_id, input_dir, output_dir, categories)
        total_converted += converted
    
    print(f"\nDone! Converted {total_converted} images across {len(scene_ids)} scenes")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
