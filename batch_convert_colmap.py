#!/usr/bin/env python3
"""
Batch convert COLMAP outputs to training format for depth estimation
Converts all sparse/0 directories to the required bag_X/sequence_Y/ structure
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def read_colmap_cameras(cameras_file):
    """Read COLMAP cameras.txt and extract intrinsics"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]

            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_colmap_images(images_file):
    """Read COLMAP images.txt and extract poses"""
    images = {}
    with open(images_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # images.txt has alternating lines: image info, then points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        # Quaternion: qw, qx, qy, qz
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # Translation
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        image_name = parts[9]

        images[image_id] = {
            'name': image_name,
            'camera_id': camera_id,
            'quat': [qw, qx, qy, qz],
            'trans': [tx, ty, tz]
        }

    return images

def read_colmap_points3D(points_file):
    """Read COLMAP points3D.txt"""
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # X, Y, Z, R, G, B, ERROR, TRACK[]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append([x, y, z, r, g, b])

    return np.array(points)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix"""
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    return R

def convert_colmap_to_training_format(sparse_dir, output_dir, copy_images=True):
    """
    Convert a single COLMAP sparse reconstruction to training format

    Args:
        sparse_dir: Path to COLMAP sparse/0 directory
        output_dir: Path to output sequence directory
        copy_images: Whether to copy images (True) or create symlinks (False)
    """
    sparse_path = Path(sparse_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read COLMAP files
    cameras = read_colmap_cameras(sparse_path / 'cameras.txt')
    images = read_colmap_images(sparse_path / 'images.txt')
    points3D = read_colmap_points3D(sparse_path / 'points3D.txt')

    # Find original image directory (parent of sparse)
    image_source_dir = sparse_path.parent.parent

    # Create output directories
    image_dir = output_path / 'image_0'
    intrinsics_dir = output_path / 'camera_intrinsics_per_view'
    image_dir.mkdir(exist_ok=True)
    intrinsics_dir.mkdir(exist_ok=True)

    # Sort images by name (assumes sequential naming)
    sorted_images = sorted(images.items(), key=lambda x: x[1]['name'])

    # Prepare motion data (poses)
    motion_data = {}

    for idx, (img_id, img_data) in enumerate(sorted_images):
        img_name = img_data['name']
        camera_id = img_data['camera_id']

        # Copy/link image
        src_image = image_source_dir / img_name
        if not src_image.exists():
            # Try to find in subdirectories
            for root, dirs, files in os.walk(image_source_dir):
                if img_name in files:
                    src_image = Path(root) / img_name
                    break

        if src_image.exists():
            dst_image = image_dir / f'{idx:010d}.png'
            if copy_images:
                shutil.copy2(src_image, dst_image)
            else:
                if not dst_image.exists():
                    # Use absolute paths for symlinks
                    os.symlink(src_image.absolute(), dst_image)
        else:
            print(f"Warning: Source image not found: {img_name}")

        # Write camera intrinsics
        cam = cameras[camera_id]
        intrinsic_file = intrinsics_dir / f'{idx:010d}.txt'

        if cam['model'] == 'OPENCV':
            # OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
            fx, fy, cx, cy = cam['params'][0], cam['params'][1], cam['params'][2], cam['params'][3]
        elif cam['model'] == 'PINHOLE':
            # PINHOLE model: fx, fy, cx, cy
            fx, fy, cx, cy = cam['params'][0], cam['params'][1], cam['params'][2], cam['params'][3]
        else:
            # Default: assume first 4 params are fx, fy, cx, cy
            fx, fy, cx, cy = cam['params'][0], cam['params'][1], cam['params'][2], cam['params'][3]

        # Write 3x3 intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        np.savetxt(intrinsic_file, K, fmt='%.6f')

        # Convert COLMAP pose to motion format
        # COLMAP uses world-to-camera transformation
        # We need camera-to-world for motion.yaml
        R_w2c = quaternion_to_rotation_matrix(img_data['quat'])
        t_w2c = np.array(img_data['trans']).reshape(3, 1)

        # Invert to get camera-to-world
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        # Store as 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_c2w
        T[:3, 3] = t_c2w.flatten()

        motion_data[idx] = T.tolist()

    # Write motion.yaml
    motion_file = output_path / 'motion.yaml'
    with open(motion_file, 'w') as f:
        yaml.dump({'motion': motion_data}, f)

    # Write structure.ply (sparse 3D points)
    ply_file = output_path / 'structure.ply'
    with open(ply_file, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points3D)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for point in points3D:
            x, y, z, r, g, b = point
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n')

    return len(sorted_images), len(points3D)

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_convert_colmap.py <dataset_root> <output_root>")
        print("Example: python batch_convert_colmap.py /home/test1/workspace/densnet/trainin_data/dataset /home/test1/workspace/training_formatted")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not dataset_root.exists():
        print(f"ERROR: Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # Find all sparse/0 directories
    print("Scanning for COLMAP reconstructions...")
    sparse_dirs = list(dataset_root.rglob('sparse/0'))

    if not sparse_dirs:
        print(f"ERROR: No sparse/0 directories found in {dataset_root}")
        sys.exit(1)

    print(f"Found {len(sparse_dirs)} COLMAP reconstructions")
    print()

    # Convert each one
    bag_idx = 0
    seq_idx = 0
    success = 0
    failed = 0

    log_file = output_root / 'conversion_log.txt'

    with open(log_file, 'w') as log:
        log.write(f"Conversion started\n")
        log.write(f"Dataset root: {dataset_root}\n")
        log.write(f"Output root: {output_root}\n\n")

        for sparse_dir in tqdm(sparse_dirs, desc="Converting"):
            try:
                # Create output directory: bag_X/sequence_Y
                output_seq_dir = output_root / f'bag_{bag_idx}' / f'sequence_{seq_idx}'

                # Get relative path for logging
                rel_path = sparse_dir.relative_to(dataset_root)

                # Convert
                num_images, num_points = convert_colmap_to_training_format(
                    sparse_dir, output_seq_dir, copy_images=False
                )

                log_msg = f"✓ {rel_path} -> bag_{bag_idx}/sequence_{seq_idx} ({num_images} images, {num_points} points)\n"
                log.write(log_msg)
                success += 1

                # Increment sequence/bag counter
                seq_idx += 1
                if seq_idx >= 10:  # 10 sequences per bag
                    seq_idx = 0
                    bag_idx += 1

            except Exception as e:
                log_msg = f"✗ {rel_path}: {str(e)}\n"
                log.write(log_msg)
                print(f"\nError processing {sparse_dir}: {e}")
                failed += 1

    # Summary
    print()
    print("="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Total: {len(sparse_dirs)}")
    print(f"  ✓ Success: {success}")
    print(f"  ✗ Failed: {failed}")
    print()
    print(f"Output directory: {output_root}")
    print(f"Log file: {log_file}")
    print()
    print("Next step: Update config file and start training!")

if __name__ == '__main__':
    main()
