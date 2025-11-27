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
# Assuming you have bags organized as bag_0, bag_1, ..., bag_N
# Adjust these ranges based on your actual number of bags

# Example: If you have 225 sequences organized into bags (10 seq/bag = ~23 bags)
# Training: bags 0-17 (80%)
# Validation: bags 18-20 (10%)
# Testing: bags 21-22 (10%)

TRAINING_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"
VALIDATION_IDS="18 19 20"
TESTING_IDS="21 22"

# ============================================
# TRAINING PARAMETERS
# ============================================
INPUT_SIZE="320 192"  # Width Height (must be divisible by 64)
ADJACENT_RANGE="1 2 3"  # Frame intervals for pairs
ID_RANGE="0 10000"  # Frame ID range (adjust based on your data)
BATCH_SIZE=8
NUM_WORKERS=4
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
    --number_epoch $NUM_EPOCHS \
    --num_iter $NUM_ITER \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --dcl_weight $DCL_WEIGHT \
    --sfl_weight $SFL_WEIGHT \
    --device auto

echo ""
echo "Training complete!"
