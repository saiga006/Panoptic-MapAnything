#!/usr/bin/env python3
"""
Check which scenes in the train subset file exist in the dataset.
Identifies missing/corrupted scenes before running expensive rasterization.
Copies all valid scenes to a separate folder.
"""

import os
import sys
import shutil
from pathlib import Path

def check_scene_completeness(scene_id, data_root):
    """Check if a scene has all required files for rasterization."""
    scene_path = Path(data_root) / scene_id
    
    issues = []
    
    # Check scene directory exists
    if not scene_path.exists():
        return False, ["Scene directory does not exist"]
    
    # Check mesh file
    mesh_path = scene_path / "scans" / "mesh_aligned_0.05.ply"
    if not mesh_path.exists():
        issues.append(f"Missing mesh: {mesh_path}")
    
    # Check DSLR camera data
    dslr_colmap = scene_path / "dslr" / "colmap"
    if not dslr_colmap.exists():
        issues.append(f"Missing DSLR colmap directory: {dslr_colmap}")
    else:
        cameras_file = dslr_colmap / "cameras.txt"
        images_file = dslr_colmap / "images.txt"
        if not cameras_file.exists():
            issues.append(f"Missing cameras.txt: {cameras_file}")
        if not images_file.exists():
            issues.append(f"Missing images.txt: {images_file}")
    
    # Check DSLR images directory
    dslr_images = scene_path / "dslr" / "undistorted_images"
    if not dslr_images.exists():
        # Try resized_images as fallback
        dslr_images = scene_path / "dslr" / "resized_images"
        if not dslr_images.exists():
            issues.append(f"Missing DSLR images directory")
    
    return len(issues) == 0, issues


def main():
    # Paths
    data_root = "/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/datasets/scannet/scannetpp/data"
    scene_list_file = "/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/datasets/scannet/scannetpp/splits/nvs_sem_train_subset150.txt"
    
    print("=" * 80)
    print("ScanNet++ Scene Validation")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Scene list: {scene_list_file}")
    print("=" * 80)
    print()
    
    # Read scene list
    with open(scene_list_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Total scenes to check: {len(scene_ids)}")
    print()
    
    # Check each scene
    valid_scenes = []
    broken_scenes = []
    
    for i, scene_id in enumerate(scene_ids, 1):
        is_valid, issues = check_scene_completeness(scene_id, data_root)
        
        if is_valid:
            valid_scenes.append(scene_id)
            status = "✓"
        else:
            broken_scenes.append((scene_id, issues))
            status = "✗"
        
        # Print progress every 10 scenes
        if i % 10 == 0 or not is_valid:
            print(f"[{i}/{len(scene_ids)}] {status} {scene_id}")
            if not is_valid:
                for issue in issues:
                    print(f"    - {issue}")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Valid scenes:  {len(valid_scenes)}/{len(scene_ids)}")
    print(f"✗ Broken scenes: {len(broken_scenes)}/{len(scene_ids)}")
    print()
    
    if broken_scenes:
        print("Broken scenes:")
        for scene_id, issues in broken_scenes:
            print(f"  - {scene_id}")
            for issue in issues:
                print(f"      {issue}")
        print()
        
        # Create cleaned scene list
        output_file = scene_list_file.replace(".txt", "_clean.txt")
        with open(output_file, 'w') as f:
            for scene_id in valid_scenes:
                f.write(f"{scene_id}\n")
        
        print(f"✓ Created cleaned scene list: {output_file}")
        print(f"  ({len(valid_scenes)} valid scenes)")
        print()
        print("To use the cleaned list, update your config:")
        print(f"  scene_list_file: .../{Path(output_file).name}")
    else:
        print("✓ All scenes are valid!")
    
    # Copy valid scenes to separate folder
    if valid_scenes:
        valid_scenes_dir = Path(data_root) / "valid_scenes"
        print()
        print("=" * 80)
        print("COPYING VALID SCENES")
        print("=" * 80)
        print(f"Destination: {valid_scenes_dir}")
        print(f"Scenes to copy: {len(valid_scenes)}")
        print()
        
        valid_scenes_dir.mkdir(exist_ok=True)
        
        for i, scene_id in enumerate(valid_scenes, 1):
            src_path = Path(data_root) / scene_id
            dst_path = valid_scenes_dir / scene_id
            
            try:
                if dst_path.exists():
                    print(f"[{i}/{len(valid_scenes)}] ⊙ {scene_id} (already exists, skipping)")
                else:
                    print(f"[{i}/{len(valid_scenes)}] → {scene_id} (copying...)")
                    shutil.copytree(src_path, dst_path, symlinks=True)
                    print(f"[{i}/{len(valid_scenes)}] ✓ {scene_id} (completed)")
            except Exception as e:
                print(f"[{i}/{len(valid_scenes)}] ✗ {scene_id} (failed: {e})")
        
        print()
        print(f"✓ Valid scenes copied to: {valid_scenes_dir}")
        print()
    
    print("=" * 80)
    
    return len(broken_scenes) > 0


if __name__ == "__main__":
    has_errors = main()
    sys.exit(1 if has_errors else 0)
