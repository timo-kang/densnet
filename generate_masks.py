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
    """Create mask based on first image dimensions"""
    seq_path = Path(seq_dir)

    # Find first image to get dimensions
    image_dir = seq_path / 'image_0'
    images = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))

    if not images:
        return False

    # Read first image to get size
    img = cv2.imread(str(images[0]))
    if img is None:
        return False

    height, width = img.shape[:2]

    # Create white mask (all pixels valid)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Optional: Create circular mask for endoscopy (uncomment if needed)
    # center_x, center_y = width // 2, height // 2
    # radius = min(width, height) // 2 - 10
    # cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Save mask
    mask_path = seq_path / 'undistorted_mask.bmp'
    cv2.imwrite(str(mask_path), mask)

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
