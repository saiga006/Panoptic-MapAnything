#!/usr/bin/env python3
"""
Create a subset ground truth JSON for testing evaluation with fewer images.
This allows testing the evaluation pipeline without processing all 5,000 images.
"""

import json
import argparse
from pathlib import Path

def create_subset_gt(input_json, output_json, num_images):
    """
    Create a subset of the ground truth JSON with only the first N images
    
    Args:
        input_json: Path to full ground truth JSON
        output_json: Path to save subset JSON
        num_images: Number of images to include
    """
    print(f"Loading ground truth from: {input_json}")
    with open(input_json, 'r') as f:
        gt_data = json.load(f)
    
    print(f"Original dataset:")
    print(f"  Images: {len(gt_data['images'])}")
    print(f"  Annotations: {len(gt_data['annotations'])}")
    print(f"  Categories: {len(gt_data['categories'])}")
    
    # Get first N images
    subset_images = gt_data['images'][:num_images]
    image_ids = {img['id'] for img in subset_images}
    
    # Filter annotations to only those for subset images
    subset_annotations = [
        ann for ann in gt_data['annotations'] 
        if ann['image_id'] in image_ids
    ]
    
    # Create new subset JSON
    subset_data = {
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': gt_data['categories']
    }
    
    print(f"\nSubset dataset:")
    print(f"  Images: {len(subset_data['images'])}")
    print(f"  Annotations: {len(subset_data['annotations'])}")
    print(f"  Categories: {len(subset_data['categories'])}")
    
    # Save subset
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(subset_data, f)
    
    print(f"\n✓ Subset ground truth saved to: {output_json}")
    print(f"\nTo evaluate with this subset:")
    print(f"  python -m panopticapi.evaluation \\")
    print(f"    --gt_json_file {output_json} \\")
    print(f"    --gt_folder ./datasets/coco/panoptic_val2017 \\")
    print(f"    --pred_json_file ./inference_coco_val/coco_val_panoptic_predictions.json \\")
    print(f"    --pred_folder ./inference_coco_val/panoptic_predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create subset ground truth JSON for testing"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./datasets/coco/annotations/panoptic_val2017.json",
        help="Path to full ground truth JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/coco/annotations/panoptic_val2017_subset.json",
        help="Path to save subset JSON"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to include in subset"
    )
    
    args = parser.parse_args()
    
    create_subset_gt(args.input, args.output, args.num_images)
