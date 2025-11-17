# Complete Workflow: Spine Endoscopy Depth Estimation

## Overview
This guide walks you through the complete pipeline from raw 640×360 spine endoscopy images to trained depth estimation model.

---

## Step 1: Prepare Your Images

**What you need:**
- Video frames extracted as images: `frame_0001.jpg, frame_0002.jpg, ...`
- Image size: 640×360 pixels (or similar)
- Sequential frames from endoscopy video

**Organize them:**
```
/data/spine_sequence_1/
├── frame_0001.jpg
├── frame_0002.jpg
├── frame_0003.jpg
└── ...
```

---

## Step 2: Run Structure from Motion (SfM)

**What it does:**
- Analyzes your image sequence
- Figures out camera movement (position + orientation)
- Creates sparse 3D reconstruction
- Extracts camera calibration

**How to run:**

### Option A: Using COLMAP (Recommended)

1. **Install COLMAP:**
   ```bash
   # macOS
   brew install colmap

   # Ubuntu
   sudo apt install colmap

   # Or download from: https://colmap.github.io/
   ```

2. **Edit and run the script:**
   ```bash
   # Edit paths in the script
   nano run_colmap_spine.sh

   # Run COLMAP
   ./run_colmap_spine.sh
   ```

3. **Check reconstruction quality:**
   ```bash
   colmap gui --import_path /path/to/colmap/workspace/sparse/0
   ```

   **What to look for:**
   - Camera trajectory should be smooth
   - Sparse point cloud should cover the visible anatomy
   - Most frames should be registered (green cameras in GUI)

**Expected outputs:**
```
/colmap/workspace/sparse/0/
├── cameras.txt      # Camera intrinsics
├── images.txt       # Camera poses
├── points3D.txt     # Sparse 3D points
└── points3D.ply     # PLY format
```

---

## Step 3: Convert to Required Format

**Run the conversion script:**
```bash
python convert_colmap_to_required_format.py \
    --colmap_dir /path/to/colmap/workspace/sparse/0 \
    --output_dir /data/training_data/bag_1/sequence_1 \
    --image_dir /data/spine_sequence_1
```

**This generates:**
```
/data/training_data/bag_1/sequence_1/
├── 00000001.jpg                    # Images
├── 00000002.jpg
├── camera_intrinsics_per_view      # Camera calibration
├── motion.yaml                     # Camera poses
├── structure.ply                   # Sparse 3D points
├── view_indexes_per_point          # Point visibility
├── visible_view_indexes            # Registered frames
├── selected_indexes                # Frame indices
└── undistorted_mask.bmp           # Valid region mask
```

**Organize for training:**
```
/data/training_data/
├── bag_1/                          # Patient/session 1
│   ├── sequence_1/                 # Video sequence 1
│   └── sequence_2/                 # Video sequence 2 (optional)
├── bag_2/                          # Patient/session 2
│   └── sequence_1/
└── bag_3/                          # Patient/session 3
    └── sequence_1/
```

---

## Step 4: Train the Model

**Edit training script:**
```bash
nano train_spine.sh
```

**Update these lines:**
```bash
TRAINING_DATA_ROOT="/data/training_data"
TRAINING_RESULT_ROOT="/data/results"

# Split your data
TRAINING_PATIENT_ID="1 2"      # For training
VALIDATION_PATIENT_ID="3"      # For validation
TESTING_PATIENT_ID="4"         # For testing

ID_RANGE="1 4"                 # Total number of bags
```

**Run training:**
```bash
./train_spine.sh
```

**Monitor progress:**
```bash
tensorboard --logdir=/data/results
# Open browser to http://localhost:6006
```

**Training time:**
- ~10-20 hours for 100 epochs on a good GPU (RTX 3090)
- ~30-50 hours on moderate GPU (GTX 1080)

**What to monitor:**
- `loss_sparse_flow`: Should decrease to ~0.01-0.05
- `loss_depth_consistency`: Should decrease to ~0.1-0.3
- Depth visualizations should show reasonable structure

---

## Step 5: Evaluate the Model

**After training completes:**
```bash
# Edit evaluation script
nano evaluate_spine.sh

# Update model path
TRAINED_MODEL_PATH="/data/results/checkpoint_model_epoch_100_validation_0.05.pt"

# Run evaluation
./evaluate_spine.sh
```

**Outputs:**
- Predicted depth maps for test sequences
- Dense 3D point clouds
- Quantitative metrics (if ground truth available)

---

## Troubleshooting

### SfM Fails / Few Frames Registered
**Problem:** COLMAP only registers 10% of frames
**Solutions:**
- Ensure images have good texture (not too blurry)
- Check lighting consistency
- Use `--ImageReader.camera_model OPENCV` to handle distortion
- Increase `--SiftExtraction.max_num_features 8192`

### Training Loss is NaN
**Problem:** Loss becomes NaN during training
**Solutions:**
- Lower learning rate: `--max_lr 5.0e-4`
- Reduce batch size: `--batch_size 4`
- Check SfM quality (sparse points should be reasonable)

### GPU Out of Memory
**Problem:** CUDA out of memory error
**Solutions:**
- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Reduce input size: `--input_size 256 128`
- Use single GPU: `CUDA_VISIBLE_DEVICES=0 python train.py ...`

### Poor Depth Quality
**Problem:** Predicted depths look wrong
**Solutions:**
- Train longer (100-200 epochs)
- Ensure good SfM reconstruction quality
- Check camera motion diversity (not just forward/backward)
- Increase `--visibility_overlap 50` for more stable training

---

## Quick Validation Checklist

Before full training, verify:

- [ ] SfM reconstruction looks reasonable in COLMAP GUI
- [ ] At least 70% of frames are registered
- [ ] Sparse point cloud covers the anatomy
- [ ] All required files are generated
- [ ] Image sizes are divisible by 64 after preprocessing
- [ ] Can run 1 epoch without errors (use test command below)

**Test command:**
```bash
python train.py \
    --training_data_root "./example_training_data_root" \
    --training_result_root "./test_output" \
    --input_size 256 320 \
    --input_downsampling 4.0 \
    --network_downsampling 64 \
    --batch_size 2 \
    --num_workers 2 \
    --num_pre_workers 2 \
    --number_epoch 1 \
    --num_iter 10 \
    --adjacent_range 5 30 \
    --dcl_weight 5.0 \
    --sfl_weight 20.0 \
    --max_lr 1.0e-3 \
    --min_lr 1.0e-4 \
    --training_patient_id 1 \
    --validation_patient_id 1 \
    --testing_patient_id 1 \
    --id_range 1 2 \
    --inlier_percentage 0.99 \
    --visibility_overlap 30 \
    --validation_interval 1 \
    --display_interval 5 \
    --architecture_summary
```

---

## Summary: What Does Each Tool Do?

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| **COLMAP** | Image sequence | 3D points + camera poses | Recover geometry and motion |
| **convert_colmap.py** | COLMAP files | Required format | Format conversion |
| **train.py** | Formatted data | Trained model | Learn dense depth |
| **evaluate.py** | Trained model + test data | Depth maps | Generate predictions |

**Key insight:** SfM provides the "supervision signal" that allows self-supervised depth learning without ground truth depth labels!
