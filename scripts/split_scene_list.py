#!/usr/bin/env python3
"""
Split scene list into batches for parallel/resumable processing.

Creates multiple scene list files that can be processed independently,
allowing you to:
1. Process scenes in parallel across multiple jobs
2. Resume from any batch if interrupted

Usage:
    python split_scene_list.py \
        --input nvs_sem_train_subset150.txt \
        --output_prefix train_batch \
        --scenes_per_batch 30
        
This will create: train_batch_001.txt, train_batch_002.txt, etc.
"""

import argparse
from pathlib import Path


def split_scene_list(input_file: str, output_prefix: str, scenes_per_batch: int = 30):
    """Split scene list into batches."""
    
    # Read input file
    with open(input_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    
    print(f"Total scenes: {len(scenes)}")
    print(f"Scenes per batch: {scenes_per_batch}")
    
    num_batches = (len(scenes) + scenes_per_batch - 1) // scenes_per_batch
    print(f"Number of batches: {num_batches}")
    
    input_path = Path(input_file)
    output_dir = input_path.parent
    
    batch_files = []
    
    for i in range(num_batches):
        start_idx = i * scenes_per_batch
        end_idx = min((i + 1) * scenes_per_batch, len(scenes))
        batch_scenes = scenes[start_idx:end_idx]
        
        # Create batch file
        batch_num = i + 1
        output_file = output_dir / f"{output_prefix}_{batch_num:03d}.txt"
        
        with open(output_file, 'w') as f:
            for scene in batch_scenes:
                f.write(f"{scene}\n")
        
        batch_files.append(output_file)
        print(f"  Batch {batch_num}: {len(batch_scenes)} scenes → {output_file.name}")
    
    print(f"\nCreated {len(batch_files)} batch files")
    print("\nTo process:")
    print("  1. Update scene_list_file in your config to use batch file")
    print("  2. Submit separate jobs for each batch")
    print("  3. Or process batches sequentially - if interrupted, restart from next batch")
    
    return batch_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split scene list into batches')
    parser.add_argument('--input', type=str, required=True,
                        help='Input scene list file')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output batch files')
    parser.add_argument('--scenes_per_batch', type=int, default=30,
                        help='Number of scenes per batch (default: 30)')
    args = parser.parse_args()
    
    split_scene_list(args.input, args.output_prefix, args.scenes_per_batch)
