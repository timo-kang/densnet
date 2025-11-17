#!/usr/bin/env python3
"""
Script to automatically fix hardcoded .cuda() calls in models.py
Replaces them with device-agnostic code
"""

import re
from pathlib import Path

def fix_models_cuda_calls(file_path):
    """Fix all .cuda() calls in models.py"""

    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern 1: torch.tensor(...).float().cuda() -> tensor with device
    # We'll need to ensure device is available in context

    # Fix specific patterns in _warp_coordinate_generate
    content = re.sub(
        r'torch\.arange\(start=0, end=height, dtype=torch\.float32\)\.cuda\(\)',
        r'torch.arange(start=0, end=height, dtype=torch.float32, device=device)',
        content
    )

    content = re.sub(
        r'torch\.arange\(start=0, end=width, dtype=torch\.float32\)\.cuda\(\)',
        r'torch.arange(start=0, end=width, dtype=torch.float32, device=device)',
        content
    )

    content = re.sub(
        r'torch\.ones\(\(1, height, width, 1\), dtype=torch\.float32\)\.cuda\(\)',
        r'torch.ones((1, height, width, 1), dtype=torch.float32, device=device)',
        content
    )

    content = re.sub(
        r'torch\.eye\(3\)\.float\(\)\.cuda\(\)\.reshape',
        r'torch.eye(3, dtype=torch.float32, device=device).reshape',
        content
    )

    content = re.sub(
        r'torch\.tensor\(1\.0e30\)\.float\(\)\.cuda\(\)',
        r'torch.tensor(1.0e30, dtype=torch.float32, device=device)',
        content
    )

    content = re.sub(
        r'torch\.tensor\(1\.0\)\.float\(\)\.cuda\(\)',
        r'torch.tensor(1.0, dtype=torch.float32, device=device)',
        content
    )

    content = re.sub(
        r'torch\.tensor\(0\.0\)\.float\(\)\.cuda\(\)',
        r'torch.tensor(0.0, dtype=torch.float32, device=device)',
        content
    )

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Fixed .cuda() calls in {file_path}")


if __name__ == '__main__':
    models_path = Path(__file__).parent / 'models.py'
    fix_models_cuda_calls(models_path)
    print("Done!")
