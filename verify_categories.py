#!/usr/bin/env python3
"""
Verify category ID mapping is working correctly
"""
import sys
sys.path.insert(0, './Mask2Former')

from detectron2.data import MetadataCatalog
from m2f_train_cluster_working import register_coco_panoptic

# Register COCO dataset
coco_root = "./datasets/coco"
register_coco_panoptic(coco_root)

# Get metadata
metadata = MetadataCatalog.get("coco_2017_val_panoptic")

print("="*80)
print("COCO Panoptic Category Mapping Verification")
print("="*80)

# Check if mappings exist
if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
    print(f"\n✓ thing_dataset_id_to_contiguous_id exists")
    print(f"  Length: {len(metadata.thing_dataset_id_to_contiguous_id)}")
    print(f"  Sample (first 10): {dict(list(metadata.thing_dataset_id_to_contiguous_id.items())[:10])}")
else:
    print("\n✗ thing_dataset_id_to_contiguous_id NOT FOUND")

if hasattr(metadata, 'stuff_dataset_id_to_contiguous_id'):
    print(f"\n✓ stuff_dataset_id_to_contiguous_id exists")
    print(f"  Length: {len(metadata.stuff_dataset_id_to_contiguous_id)}")
    print(f"  Sample (first 10): {dict(list(metadata.stuff_dataset_id_to_contiguous_id.items())[:10])}")
else:
    print("\n✗ stuff_dataset_id_to_contiguous_id NOT FOUND")

# Build reverse mapping
if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
    thing_contiguous_to_dataset = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
else:
    thing_contiguous_to_dataset = {}

if hasattr(metadata, 'stuff_dataset_id_to_contiguous_id'):
    stuff_contiguous_to_dataset = {v: k for k, v in metadata.stuff_dataset_id_to_contiguous_id.items()}
else:
    stuff_contiguous_to_dataset = {}

contiguous_to_dataset = {**thing_contiguous_to_dataset, **stuff_contiguous_to_dataset}

print(f"\n" + "="*80)
print(f"Reverse Mapping (Contiguous ID -> COCO ID)")
print(f"="*80)
print(f"Total mappings: {len(contiguous_to_dataset)}")
print(f"\nFirst 20 mappings:")
for cont_id in sorted(contiguous_to_dataset.keys())[:20]:
    coco_id = contiguous_to_dataset[cont_id]
    print(f"  Contiguous {cont_id:3d} -> COCO {coco_id:3d}")

print(f"\nLast 10 mappings:")
for cont_id in sorted(contiguous_to_dataset.keys())[-10:]:
    coco_id = contiguous_to_dataset[cont_id]
    print(f"  Contiguous {cont_id:3d} -> COCO {coco_id:3d}")

# Check for problematic IDs mentioned in error
print(f"\n" + "="*80)
print(f"Checking Problematic IDs from Error")
print(f"="*80)
problematic = [0, 26, 101, 111, 120, 126, 127, 129]
for cont_id in problematic:
    if cont_id in contiguous_to_dataset:
        print(f"  Contiguous {cont_id:3d} -> COCO {contiguous_to_dataset[cont_id]:3d} ✓")
    else:
        print(f"  Contiguous {cont_id:3d} -> NOT MAPPED ✗")

print(f"\n" + "="*80)
