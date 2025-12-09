#!/usr/bin/env python3
"""
Preprocess endoscopy images to enhance feature detection for COLMAP
Applies contrast enhancement, sharpening, and denoising
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def enhance_for_colmap(image):
    """
    Enhance endoscopy image to make SIFT features more detectable

    Args:
        image: Input BGR image

    Returns:
        Enhanced BGR image
    """
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    # Sharpen to enhance edges (important for SIFT)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Ensure values are in valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced

def preprocess_directory(input_dir, output_dir, upscale=False):
    """
    Preprocess all images in a directory

    Args:
        input_dir: Directory with original images
        output_dir: Where to save enhanced images
        upscale: If True, upscale images 2x for more features
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    if not images:
        return 0

    processed = 0
    for img_file in images:
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Upscale if requested (helps with feature detection)
        if upscale:
            height, width = img.shape[:2]
            img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        # Enhance
        enhanced = enhance_for_colmap(img)

        # Save
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed += 1

    return processed

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess_endoscopy_images.py <input_base_dir> <output_base_dir> [--upscale]")
        print()
        print("This will enhance all images in subdirectories for better COLMAP reconstruction")
        print("Enhancements: CLAHE contrast, bilateral denoising, sharpening")
        print()
        print("Options:")
        print("  --upscale: Upscale images 2x (640x360 → 1280x720) for more features")
        print()
        print("Example:")
        print("  python preprocess_endoscopy_images.py \\")
        print("    /home/test1/dataset/extracted_sequences_25frames \\")
        print("    /home/test1/dataset/extracted_sequences_enhanced \\")
        print("    --upscale")
        sys.exit(1)

    input_base = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    upscale = '--upscale' in sys.argv

    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        sys.exit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    print("=== Endoscopy Image Enhancement for COLMAP ===")
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")
    print(f"Upscale 2x: {upscale}")
    print()

    # Find all subdirectories with images
    subdirs = []
    for item in input_base.iterdir():
        if item.is_dir():
            if list(item.glob('*.jpg')) or list(item.glob('*.png')):
                subdirs.append(item)

    print(f"Found {len(subdirs)} directories with images")
    print()

    total_processed = 0
    log_file = output_base / 'preprocessing_log.txt'

    with open(log_file, 'w') as log:
        log.write(f"Input: {input_base}\n")
        log.write(f"Output: {output_base}\n")
        log.write(f"Upscale: {upscale}\n\n")

        for subdir in tqdm(subdirs, desc="Processing directories"):
            output_subdir = output_base / subdir.name

            processed = preprocess_directory(subdir, output_subdir, upscale=upscale)

            if processed > 0:
                log.write(f"✓ {subdir.name}: {processed} images\n")
                total_processed += processed
            else:
                log.write(f"✗ {subdir.name}: No images found\n")

    print()
    print("=== Enhancement Complete ===")
    print(f"Total images processed: {total_processed}")
    print(f"Log: {log_file}")
    print()
    print("Next step: Run COLMAP on enhanced images")
    print(f"  ./batch_process_colmap.sh {output_base} <colmap_output>")

if __name__ == '__main__':
    main()
