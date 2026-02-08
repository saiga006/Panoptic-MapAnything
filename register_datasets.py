#!/usr/bin/env python3
"""
Register ScanNet++ datasets before training.

This script registers the panoptic datasets that match the names in the config:
- scannetpp_panoptic_train
- scannetpp_panoptic_val

Usage:
    python register_datasets.py
"""

import os
import sys
from pathlib import Path

# Add Mask2Former to path
sys.path.insert(0, str(Path(__file__).parent))

from detectron2.data import DatasetCatalog, MetadataCatalog
from mask2former.data.dataset_mappers.scannetpp_panoptic_dataset_mapper import (
    _load_scannetpp_panoptic_dataset
)


def register_scannetpp_train_val():
    """Register training and validation datasets."""
    
    # Paths (adjust if needed)
    default_root = "/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/datasets/scannet/scannetpp"
    scannetpp_root = os.environ.get("SCANNETPP_ROOT", default_root)
    
    panoptic_train = os.environ.get("PANOPTIC_ROOT", f"{scannetpp_root}/panoptic")
    panoptic_val = os.environ.get("PANOPTIC_VAL_ROOT", f"{scannetpp_root}/panoptic_val")
    splits_dir = os.environ.get("SPLIT_DIR", f"{scannetpp_root}/splits")
    
    print(f"Registering datasets from: {scannetpp_root}")
    
    # Training set
    train_split = f"{splits_dir}/nvs_sem_train_subset150.txt"  # Using your 150-scene subset
    if not os.path.exists(train_split):
        # Fallback to standard split if subset not found
        train_split = f"{splits_dir}/nvs_sem_train.txt"

    if os.path.exists(train_split):
        DatasetCatalog.register(
            "scannetpp_panoptic_train",
            lambda: _load_scannetpp_panoptic_dataset(
                data_root=scannetpp_root,
                split_file=train_split,
                panoptic_dir=panoptic_train,
                image_type="dslr",
                use_undistorted=True,
            )
        )
        
        MetadataCatalog.get("scannetpp_panoptic_train").set(
            panoptic_root=panoptic_train,
            image_root=scannetpp_root,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=10000,  # CRITICAL: Must match convert_to_panoptic_format.py
            thing_dataset_id_to_contiguous_id={},  # Will be populated by dataset
            stuff_dataset_id_to_contiguous_id={},  # Required by COCOPanopticEvaluator
        )
        print(f"✓ Registered: scannetpp_panoptic_train ({train_split})")
    else:
        print(f"✗ Training split not found: {train_split}")
    
    # Validation set
    val_split = f"{splits_dir}/nvs_sem_val_clean.txt"
    if os.path.exists(val_split):
        DatasetCatalog.register(
            "scannetpp_panoptic_val",
            lambda: _load_scannetpp_panoptic_dataset(
                data_root=scannetpp_root,
                split_file=val_split,
                panoptic_dir=panoptic_val,
                image_type="dslr",
                use_undistorted=True,
            )
        )
        
        MetadataCatalog.get("scannetpp_panoptic_val").set(
            panoptic_root=panoptic_val,
            image_root=scannetpp_root,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            label_divisor=10000,  # CRITICAL: Must match convert_to_panoptic_format.py
            thing_dataset_id_to_contiguous_id={},
            stuff_dataset_id_to_contiguous_id={},  # Required by COCOPanopticEvaluator
        )
        print(f"✓ Registered: scannetpp_panoptic_val ({val_split})")
    else:
        print(f"✗ Validation split not found: {val_split}")


if __name__ == "__main__":
    register_scannetpp_train_val()
    
    # Verify registration
    print("\nRegistered datasets:")
    for name in ["scannetpp_panoptic_train", "scannetpp_panoptic_val"]:
        if DatasetCatalog.is_registered(name):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (NOT REGISTERED)")
