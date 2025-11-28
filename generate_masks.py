#!/usr/bin/env python3
"""
Generate undistorted_mask.bmp files for all sequences
Creates white masks (all pixels valid) for COLMAP-converted data
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def create_mask_for_sequence(seq_dir):
    """Create mask and selected_indexes based on available images"""
    seq_path = Path(seq_dir)

    # Find all images (directly in sequence directory, not in image_0 subdirectory)
    images = sorted(list(seq_path.glob('*.png')) + list(seq_path.glob('*.jpg')))

    if not images:
        return False

    # Read first image to get size
    img = cv2.imread(str(images[0]))
    if img is None:
        return False

    height, width = img.shape[:2]

    # Create white mask (all pixels valid)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Save mask
    mask_path = seq_path / 'undistorted_mask.bmp'
    cv2.imwrite(str(mask_path), mask)

    # Create selected_indexes file (sequential indices)
    selected_indexes_path = seq_path / 'selected_indexes'
    with open(selected_indexes_path, 'w') as f:
        for i in range(len(images)):
            f.write(f'{i}\n')

    # Create visible_view_indexes file (all views have visible points for COLMAP data)
    visible_view_indexes_path = seq_path / 'visible_view_indexes'
    with open(visible_view_indexes_path, 'w') as f:
        for i in range(len(images)):
            f.write(f'{i}\n')

    # Create view_indexes_per_point file
    # Format: -1 marks new point, then list view indices where it's visible
    # For COLMAP with few images, assume all points visible in all views
    view_indexes_per_point_path = seq_path / 'view_indexes_per_point'

    # Count points from structure.ply
    ply_path = seq_path / 'structure.ply'
    num_points = 0
    if ply_path.exists():
        with open(ply_path, 'r') as f:
            for line in f:
                if line.startswith('element vertex'):
                    num_points = int(line.split()[2])
                    break

    # Write view_indexes_per_point
    with open(view_indexes_per_point_path, 'w') as f:
        for point_idx in range(num_points):
            f.write('-1\n')  # Start new point
            # Assume all points visible in all views (simple approximation)
            for view_idx in range(len(images)):
                f.write(f'{view_idx}\n')

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_masks.py <training_data_root>")
        print("Example: python generate_masks.py /home/test1/workspace/training_data_formatted")
        sys.exit(1)

    data_root = Path(sys.argv[1])

    if not data_root.exists():
        print(f"ERROR: Directory not found: {data_root}")
        sys.exit(1)

    # Find all sequence directories
    print("Scanning for sequences...")
    sequences = list(data_root.glob('bag_*/sequence_*/'))

    if not sequences:
        print(f"ERROR: No bag_*/sequence_*/ directories found in {data_root}")
        sys.exit(1)

    print(f"Found {len(sequences)} sequences")
    print("Generating masks...\n")

    success = 0
    failed = 0

    for seq_dir in tqdm(sequences, desc="Creating masks"):
        if create_mask_for_sequence(seq_dir):
            success += 1
        else:
            failed += 1
            print(f"\nWarning: Failed to create mask for {seq_dir}")

    print(f"\n{'='*60}")
    print("MASK GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print()

if __name__ == '__main__':
    main()
