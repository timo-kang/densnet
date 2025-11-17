#!/bin/bash
# Evaluation script for spine endoscopy depth estimation

# Path to trained model checkpoint
TRAINED_MODEL_PATH="/path/to/checkpoint_model_epoch_100_validation_X.XX.pt"  # Update this!

# Path to test data (same format as training data)
EVALUATION_DATA_ROOT="/path/to/your/spine/test/data"  # Update this!
EVALUATION_RESULT_ROOT="/path/to/save/evaluation/results"  # Update this!

# Must match training parameters
INPUT_SIZE="320 192"
INPUT_DOWNSAMPLING=2.0
ADJACENT_RANGE="5 30"

# Testing parameters
TESTING_PATIENT_ID="4"  # Should match your test set
ID_RANGE="1 5"
BATCH_SIZE=1  # Use 1 for evaluation

# Execute evaluation
python evaluate.py \
    --trained_model_path "$TRAINED_MODEL_PATH" \
    --evaluation_data_root "$EVALUATION_DATA_ROOT" \
    --evaluation_result_root "$EVALUATION_RESULT_ROOT" \
    --input_size $INPUT_SIZE \
    --input_downsampling $INPUT_DOWNSAMPLING \
    --network_downsampling 64 \
    --batch_size $BATCH_SIZE \
    --num_workers 2 \
    --num_pre_workers 8 \
    --adjacent_range $ADJACENT_RANGE \
    --testing_patient_id $TESTING_PATIENT_ID \
    --id_range $ID_RANGE \
    --inlier_percentage 0.99 \
    --visibility_overlap 30 \
    --phase "test" \
    --load_intermediate_data \
    --architecture_summary \
    --load_all_frames

echo "Evaluation complete! Results saved to: $EVALUATION_RESULT_ROOT"
