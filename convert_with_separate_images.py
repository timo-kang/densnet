#!/usr/bin/env python3
"""
Convert COLMAP outputs to training format
Handles separate image source and COLMAP output directories
"""

import os
import sys
import yaml
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

def read_colmap_cameras(cameras_file):
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(x) for x in parts[4:]]
            }
    return cameras

def read_colmap_images(images_file):
    images = {}
    with open(images_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        images[image_id] = {
            'name': parts[9],
            'camera_id': int(parts[8]),
            'quat': [float(parts[j]) for j in range(1, 5)],
            'trans': [float(parts[j]) for j in range(5, 8)]
        }
    return images

def read_colmap_points3D(points_file):
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            points.append([float(parts[1]), float(parts[2]), float(parts[3]),
                          int(parts[4]), int(parts[5]), int(parts[6])])
    return np.array(points) if points else np.array([]).reshape(0, 6)

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    return [qw, qx, qy, qz]

def find_image(image_dir, image_name):
    """Search for image in directory tree"""
    direct = image_dir / image_name
    if direct.exists():
        return direct

    for root, _, files in os.walk(image_dir):
        if image_name in files:
            return Path(root) / image_name
    return None

def convert_sequence(sparse_dir, output_dir, image_source_dir):
    """Convert one COLMAP sequence to training format"""
    sparse_path = Path(sparse_dir)
    output_path = Path(output_dir)
    image_source_path = Path(image_source_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Read COLMAP data
    cameras = read_colmap_cameras(sparse_path / 'cameras.txt')
    images = read_colmap_images(sparse_path / 'images.txt')
    points3D = read_colmap_points3D(sparse_path / 'points3D.txt')

    # Create output structure
    img_dir = output_path / 'image_0'
    img_dir.mkdir(exist_ok=True)

    poses_list = []  # ROS-style pose messages
    intrinsics_list = []  # Collect intrinsics for single file
    found = 0
    missing = []

    sorted_imgs = sorted(images.items(), key=lambda x: x[1]['name'])

    for idx, (img_id, img_data) in enumerate(sorted_imgs):
        img_name = img_data['name']

        # Find and copy image
        src = find_image(image_source_path, img_name)
        if src:
            dst = img_dir / f'{idx:08d}.jpg'
            # Read and re-save as jpg if source is png
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), img)
                found += 1
            else:
                missing.append(img_name)
                continue
        else:
            missing.append(img_name)
            continue

        # Collect intrinsics
        cam = cameras[img_data['camera_id']]
        fx, fy, cx, cy = cam['params'][:4]
        intrinsics_list.extend([fx, fy, cx, cy])  # Add to list for single file

        # Compute pose (convert to camera-to-world)
        R_w2c = quaternion_to_rotation_matrix(img_data['quat'])
        t_w2c = np.array(img_data['trans']).reshape(3, 1)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        # Convert back to quaternion for ROS format
        quat_c2w = rotation_matrix_to_quaternion(R_c2w)

        # Create ROS-style pose message
        pose = {
            f'poses[{idx}]': {
                'position': {
                    'x': float(t_c2w[0]),
                    'y': float(t_c2w[1]),
                    'z': float(t_c2w[2])
                },
                'orientation': {
                    'x': float(quat_c2w[1]),  # qx
                    'y': float(quat_c2w[2]),  # qy
                    'z': float(quat_c2w[3]),  # qz
                    'w': float(quat_c2w[0])   # qw
                }
            }
        }
        poses_list.append(pose)

    if found == 0:
        raise ValueError(f"No images found")

    # Write single camera_intrinsics_per_view file (one parameter per line)
    with open(output_path / 'camera_intrinsics_per_view', 'w') as f:
        for param in intrinsics_list:
            f.write(f'{param:.6f}\n')

    # Save motion in ROS format
    motion_yaml = {
        'header': {
            'seq': 0,
            'stamp': 0.0,
            'frame_id': ''
        },
        'poses[]': {}
    }

    # Merge all pose dictionaries
    for pose_dict in poses_list:
        motion_yaml['poses[]'].update(pose_dict)

    with open(output_path / 'motion.yaml', 'w') as f:
        yaml.dump(motion_yaml, f, default_flow_style=False)

    # Save structure
    with open(output_path / 'structure.ply', 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(points3D)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for pt in points3D:
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(pt[3])} {int(pt[4])} {int(pt[5])}\n')

    return found, len(points3D)

def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_with_separate_images.py <colmap_dir> <image_dir> <output_dir>")
        print("\nExample:")
        print("  python convert_with_separate_images.py \\")
        print("    /home/test1/workspace/densnet/training_data/dataset \\")
        print("    /home/test1/dataset \\")
        print("    /home/test1/workspace/training_data_formatted")
        sys.exit(1)

    colmap_root = Path(sys.argv[1])
    image_root = Path(sys.argv[2])
    output_root = Path(sys.argv[3])

    if not colmap_root.exists():
        print(f"ERROR: COLMAP directory not found: {colmap_root}")
        sys.exit(1)
    if not image_root.exists():
        print(f"ERROR: Image directory not found: {image_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # Find all sparse/0 directories
    print("Scanning for COLMAP reconstructions...")
    sparse_dirs = list(colmap_root.rglob('sparse/0'))

    if not sparse_dirs:
        print(f"ERROR: No sparse/0 found in {colmap_root}")
        sys.exit(1)

    print(f"Found {len(sparse_dirs)} reconstructions")
    print(f"COLMAP: {colmap_root}")
    print(f"Images: {image_root}")
    print(f"Output: {output_root}\n")

    bag_idx = 0
    seq_idx = 0
    success = 0
    failed = 0

    log = output_root / 'conversion.log'

    with open(log, 'w') as f:
        f.write(f"COLMAP: {colmap_root}\nImages: {image_root}\nOutput: {output_root}\n\n")

        for sparse_dir in tqdm(sparse_dirs):
            try:
                # Map COLMAP path to image path
                rel = sparse_dir.parent.parent.relative_to(colmap_root)
                img_src = image_root / rel
                out_seq = output_root / f'bag_{bag_idx}' / f'sequence_{seq_idx}'

                n_imgs, n_pts = convert_sequence(sparse_dir, out_seq, img_src)

                f.write(f"✓ {rel} -> bag_{bag_idx}/seq_{seq_idx} ({n_imgs} imgs, {n_pts} pts)\n")
                success += 1

                seq_idx += 1
                if seq_idx >= 10:
                    seq_idx = 0
                    bag_idx += 1

            except Exception as e:
                f.write(f"✗ {sparse_dir.relative_to(colmap_root)}: {e}\n")
                failed += 1

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success} | Failed: {failed}")
    print(f"Output: {output_root}")
    print(f"Log: {log}\n")

if __name__ == '__main__':
    main()
