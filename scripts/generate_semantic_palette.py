#!/usr/bin/env python3
"""
Generate color palette for ScanNet++ semantic classes.
Creates a random but consistent color for each semantic class.
"""

import numpy as np
from pathlib import Path

# Number of semantic classes in ScanNet++
# You can check this from semantic_classes.txt
NUM_CLASSES = 3000  # Overestimate to be safe

# Set seed for reproducibility
np.random.seed(42)

# Generate random RGB colors
colors = np.random.randint(0, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)

# Make background (class 0) black
colors[0] = [0, 0, 0]

# Save to file
output_path = Path(__file__).parent.parent / "datasets/scannet/scannetpp/metadata/scannetpp_semantic_palette.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savetxt(output_path, colors, fmt='%d')
print(f"Created semantic color palette: {output_path}")
print(f"Generated {NUM_CLASSES} colors")
