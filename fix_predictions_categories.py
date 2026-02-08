#!/usr/bin/env python3
"""
Fix category IDs in existing predictions JSON
Converts contiguous IDs (0-132) to COCO category IDs
"""
import json
import sys
sys.path.insert(0, './Mask2Former')

from detectron2.data import MetadataCatalog
from m2f_train_cluster_working import register_coco_panoptic

# Register COCO dataset to get metadata
coco_root = "./datasets/coco"
register_coco_panoptic(coco_root)

# Get category ID mapping
metadata = MetadataCatalog.get("coco_2017_val_panoptic")

if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
    thing_contiguous_to_dataset = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
else:
    thing_contiguous_to_dataset = {}

if hasattr(metadata, 'stuff_dataset_id_to_contiguous_id'):
    stuff_contiguous_to_dataset = {v: k for k, v in metadata.stuff_dataset_id_to_contiguous_id.items()}
else:
    stuff_contiguous_to_dataset = {}

contiguous_to_dataset = {**thing_contiguous_to_dataset, **stuff_contiguous_to_dataset}

print("="*80)
print("Fixing Category IDs in Predictions")
print("="*80)
print(f"Loaded {len(contiguous_to_dataset)} category mappings\n")

# Load predictions
pred_file = "./inference_coco_val/coco_val_panoptic_predictions.json"
print(f"Loading predictions from: {pred_file}")

with open(pred_file, 'r') as f:
    predictions = json.load(f)

print(f"Found {len(predictions['annotations'])} predictions\n")

# Fix category IDs
fixed_count = 0
unmapped_ids = set()

for ann in predictions['annotations']:
    for segment in ann['segments_info']:
        contiguous_id = segment['category_id']
        
        if contiguous_id in contiguous_to_dataset:
            coco_id = contiguous_to_dataset[contiguous_id]
            segment['category_id'] = coco_id
            fixed_count += 1
        else:
            unmapped_ids.add(contiguous_id)

print(f"Fixed {fixed_count} category IDs")

if unmapped_ids:
    print(f"\n⚠ WARNING: {len(unmapped_ids)} unmapped IDs found: {sorted(unmapped_ids)}")
    print("These will keep their original IDs (may cause evaluation errors)")
else:
    print("\n✓ All category IDs successfully mapped!")

# Save fixed predictions
output_file = "./inference_coco_val/coco_val_panoptic_predictions_fixed.json"
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"\n✓ Fixed predictions saved to: {output_file}")
print("\nTo evaluate with fixed predictions:")
print("  python -m panopticapi.evaluation \\")
print("    --gt_json_file ./datasets/coco/annotations/panoptic_val2017_subset_10.json \\")
print("    --gt_folder ./datasets/coco/panoptic_val2017 \\")
print(f"    --pred_json_file {output_file} \\")
print("    --pred_folder ./inference_coco_val/panoptic_predictions")
print("="*80)
