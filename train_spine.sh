#!/bin/bash
# Training script for spine endoscopy depth estimation

# IMPORTANT: Before running this script:
# 1. Run SfM (COLMAP) on your 640×360 spine endoscopy images
# 2. Convert SfM output to required format (see README)
# 3. Organize data in folder structure like example_training_data_root/
# 4. Update paths below

# Adjust these paths
TRAINING_DATA_ROOT="/path/to/your/spine/sfm/data"  # Update this!
TRAINING_RESULT_ROOT="/path/to/save/results"       # Update this!

# Network input size (must be divisible by 64)
# For 640×360 images, recommended options:
#   - 320 192 (aspect ratio 1.67, balanced)
#   - 256 128 (aspect ratio 2.00, faster)
#   - 384 192 (aspect ratio 2.00, more detail)
INPUT_SIZE="320 192"

# Image downsampling from original to processed
# 640×360 -> intermediate processing resolution
INPUT_DOWNSAMPLING=2.0

# Training parameters
BATCH_SIZE=8
NUM_WORKERS=8
NUM_EPOCHS=100
NUM_ITER=2000  # Iterations per epoch

# Loss weights
DCL_WEIGHT=5.0   # Depth consistency loss
SFL_WEIGHT=20.0  # Sparse flow loss

# Learning rate
MAX_LR=1.0e-3
MIN_LR=1.0e-4

# Frame sampling
ADJACENT_RANGE="5 30"  # Sample frame pairs 5-30 frames apart

# Patient/sequence IDs (adjust based on your data organization)
# If you have multiple sequences, split them for train/val/test
TRAINING_PATIENT_ID="1 2"
VALIDATION_PATIENT_ID="3"
TESTING_PATIENT_ID="4"

# Data range (adjust based on your folder structure)
ID_RANGE="1 5"  # Range of bag/sequence IDs

# Execute training
python train.py \
    --training_data_root "$TRAINING_DATA_ROOT" \
    --training_result_root "$TRAINING_RESULT_ROOT" \
    --input_size $INPUT_SIZE \
    --input_downsampling $INPUT_DOWNSAMPLING \
    --network_downsampling 64 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --num_pre_workers $NUM_WORKERS \
    --number_epoch $NUM_EPOCHS \
    --num_iter $NUM_ITER \
    --adjacent_range $ADJACENT_RANGE \
    --dcl_weight $DCL_WEIGHT \
    --sfl_weight $SFL_WEIGHT \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --training_patient_id $TRAINING_PATIENT_ID \
    --validation_patient_id $VALIDATION_PATIENT_ID \
    --testing_patient_id $TESTING_PATIENT_ID \
    --id_range $ID_RANGE \
    --inlier_percentage 0.99 \
    --visibility_overlap 30 \
    --validation_interval 1 \
    --display_interval 50 \
    --architecture_summary
    # --load_intermediate_data  # Uncomment after first run for faster loading

echo "Training complete! Check results in: $TRAINING_RESULT_ROOT"
