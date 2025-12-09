#!/usr/bin/env python3
"""
Extract multiple frames from spine endoscopy videos for COLMAP reconstruction
Extracts frames at regular intervals to create temporal sequences
"""

import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, num_frames=8, start_sec=10, duration_sec=80):
    """
    Extract frames from video at regular intervals

    IMPORTANT: Use fewer frames with larger spacing for better COLMAP reconstruction!
    - 8 frames over 80 seconds = 10 second intervals
    - Larger temporal spacing = more visual change = better feature matching

    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        num_frames: How many frames to extract (default 8 for difficult endoscopy)
        start_sec: Skip first N seconds (to avoid black/unstable frames, default 10)
        duration_sec: Duration to extract from (default 80 seconds)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, "Could not open video"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration < start_sec + 5:  # Need at least 5 seconds after start_sec
        cap.release()
        return 0, f"Video too short: {duration:.1f}s"

    # Calculate frame indices to extract
    start_frame = int(start_sec * fps)
    end_frame = min(int((start_sec + duration_sec) * fps), total_frames)
    available_frames = end_frame - start_frame

    if available_frames < num_frames:
        num_frames = max(10, available_frames // 2)  # Extract at least 10 frames

    frame_interval = available_frames / num_frames
    frame_indices = [int(start_frame + i * frame_interval) for i in range(num_frames)]

    extracted = 0
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            # Save with timestamp in filename for debugging
            timestamp = frame_num / fps if fps > 0 else frame_num
            output_file = output_path / f'frame_{idx:04d}_t{timestamp:.2f}.jpg'
            cv2.imwrite(str(output_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1

    cap.release()
    return extracted, "Success"

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_video_frames.py <video_dir> <output_base_dir> [num_frames]")
        print()
        print("Example:")
        print("  python extract_video_frames.py \\")
        print("    /home/test1/dataset/patients_endoscope_videos \\")
        print("    /home/test1/dataset/extracted_sequences \\")
        print("    8")
        print()
        print("This will extract 8 frames per video from an 80-second window")
        print("LARGE spacing (10+ seconds) works better for difficult endoscopy!")
        sys.exit(1)

    video_dir = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        sys.exit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    # Find all video files
    print("Scanning for video files...")
    video_files = []
    video_files.extend(list(video_dir.rglob('*.mp4')))
    video_files.extend(list(video_dir.rglob('*.MP4')))
    video_files.extend(list(video_dir.rglob('*.avi')))
    video_files.extend(list(video_dir.rglob('*.AVI')))

    if not video_files:
        print(f"ERROR: No video files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} video files")
    print(f"Extracting {num_frames} frames per video...")
    print()

    success_count = 0
    failed_count = 0
    log_file = output_base / 'extraction_log.txt'

    with open(log_file, 'w') as log:
        log.write(f"Video frame extraction\n")
        log.write(f"Source: {video_dir}\n")
        log.write(f"Output: {output_base}\n")
        log.write(f"Frames per video: {num_frames}\n\n")

        for video_file in tqdm(video_files, desc="Extracting frames"):
            # Create output directory based on video filename
            video_name = video_file.stem
            output_dir = output_base / video_name

            # Extract frames
            extracted, message = extract_frames_from_video(
                video_file, output_dir,
                num_frames=num_frames,
                start_sec=10,  # Skip first 10 seconds (avoid unstable intro)
                duration_sec=80  # Extract from 80-second window (10s spacing)
            )

            if extracted >= 8:  # Need at least 8 frames for COLMAP
                log.write(f"✓ {video_name}: {extracted} frames\n")
                success_count += 1
            else:
                log.write(f"✗ {video_name}: {message}\n")
                failed_count += 1
                # Remove empty directory
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()

    print()
    print("="*60)
    print("FRAME EXTRACTION COMPLETE")
    print("="*60)
    print(f"Success: {success_count} videos")
    print(f"Failed: {failed_count} videos")
    print(f"Output: {output_base}")
    print(f"Log: {log_file}")
    print()
    print("Next steps:")
    print("1. Run COLMAP on extracted sequences")
    print("2. Convert to training format")
    print("3. Train depth estimation model")

if __name__ == '__main__':
    main()
