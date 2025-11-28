#!/bin/bash
# Training script for spine endoscopy depth estimation
# Adjust paths and parameters as needed

# ============================================
# PATHS - UPDATE THESE
# ============================================
DATA_ROOT="/home/test1/workspace/training_data_formatted"
RESULT_ROOT="/home/test1/workspace/training_results"

# ============================================
# DATASET SPLIT
# ============================================
# You have 51 bags total (bag_0 to bag_50)
# Split: 80% training, 10% validation, 10% testing

# Training: bags 0-39 (40 bags, ~78%)
TRAINING_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"
# Validation: bags 40-44 (5 bags, ~10%)
VALIDATION_IDS="40 41 42 43 44"
# Testing: bags 45-50 (6 bags, ~12%)
TESTING_IDS="45 46 47 48 49 50"

# ============================================
# TRAINING PARAMETERS
# ============================================
INPUT_SIZE="320 192"  # Width Height (must be divisible by 64)
ADJACENT_RANGE="1 5"  # Frame interval range [min, max] for pairs
ID_RANGE="0 10000"  # Frame ID range (adjust based on your data)
BATCH_SIZE=8
NUM_WORKERS=4
NUM_PRE_WORKERS=4
NUM_EPOCHS=50

# Learning rates
MAX_LR=1.0e-3
MIN_LR=1.0e-4

# Loss weights
DCL_WEIGHT=5.0
SFL_WEIGHT=20.0

# Iterations
NUM_ITER=100  # Iterations per epoch

# ============================================
# RUN TRAINING
# ============================================
echo "Starting training..."
echo "Data root: $DATA_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Input size: $INPUT_SIZE"
echo ""

python train.py \
    --training_data_root "$DATA_ROOT" \
    --training_result_root "$RESULT_ROOT" \
    --training_patient_id $TRAINING_IDS \
    --validation_patient_id $VALIDATION_IDS \
    --testing_patient_id $TESTING_IDS \
    --input_size $INPUT_SIZE \
    --adjacent_range $ADJACENT_RANGE \
    --id_range $ID_RANGE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --num_pre_workers $NUM_WORKERS \
    --number_epoch $NUM_EPOCHS \
    --num_iter $NUM_ITER \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --dcl_weight $DCL_WEIGHT \
    --sfl_weight $SFL_WEIGHT \
    --device auto

echo ""
echo "Training complete!"
