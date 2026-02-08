#!/usr/bin/env python3
"""
Check COCO panoptic category IDs and create a mapping
"""
import json

# Load COCO annotations
with open('./datasets/coco/annotations/panoptic_val2017.json', 'r') as f:
    coco_data = json.load(f)

print("COCO Panoptic Categories:")
print("="*80)

valid_cat_ids = set()
for cat in coco_data['categories']:
    print(f"ID: {cat['id']:3d} | Name: {cat['name']:30s} | isthing: {cat['isthing']}")
    valid_cat_ids.add(cat['id'])

print("="*80)
print(f"\nTotal categories: {len(coco_data['categories'])}")
print(f"Valid category IDs: {sorted(valid_cat_ids)[:20]}... (showing first 20)")
print(f"ID range: {min(valid_cat_ids)} to {max(valid_cat_ids)}")
print(f"\nCategories with IDs > 133: {sorted([cid for cid in valid_cat_ids if cid > 133])}")
print(f"\nMissing IDs in range 0-200:")
all_ids = set(range(0, 201))
missing = sorted(all_ids - valid_cat_ids)
print(f"Count: {len(missing)}")
print(f"Examples: {missing[:30]}")
