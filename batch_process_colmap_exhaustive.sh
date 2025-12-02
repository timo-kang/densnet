#!/bin/bash
# Batch process with EXHAUSTIVE matcher for difficult endoscopy videos
# Usage: ./batch_process_colmap_exhaustive.sh <parent_dir_with_image_folders> <output_base_dir>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <parent_dir_with_image_folders> <output_base_dir>"
    echo "Example: $0 /home/test1/dataset/extracted_test /home/test1/workspace/colmap_exhaustive"
    exit 1
fi

INPUT_BASE="$1"
OUTPUT_BASE="$2"

export QT_QPA_PLATFORM=offscreen

mkdir -p "$OUTPUT_BASE"

LOG_FILE="$OUTPUT_BASE/batch_processing.log"
PROGRESS_FILE="$OUTPUT_BASE/progress.txt"
ERROR_FILE="$OUTPUT_BASE/errors.txt"

echo "=== Batch COLMAP Processing (EXHAUSTIVE MATCHER) ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Input base: $INPUT_BASE" | tee -a "$LOG_FILE"
echo "Output base: $OUTPUT_BASE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Find all directories containing images
echo "Scanning for image directories..." | tee -a "$LOG_FILE"
DIRS=()
while IFS= read -r -d '' dir; do
    if ls "$dir"/*.jpg >/dev/null 2>&1 || ls "$dir"/*.png >/dev/null 2>&1; then
        DIRS+=("$dir")
    fi
done < <(find "$INPUT_BASE" -type d -print0)

TOTAL=${#DIRS[@]}
echo "Found $TOTAL directories with images" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $TOTAL -eq 0 ]; then
    echo "ERROR: No directories with images found!"
    exit 1
fi

SUCCESS=0
FAILED=0
SKIPPED=0

for i in "${!DIRS[@]}"; do
    INPUT_DIR="${DIRS[$i]}"
    REL_PATH="${INPUT_DIR#$INPUT_BASE/}"
    OUTPUT_DIR="$OUTPUT_BASE/$REL_PATH"
    CURRENT=$((i + 1))

    echo "======================================" | tee -a "$LOG_FILE"
    echo "Processing [$CURRENT/$TOTAL]: $REL_PATH" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"

    # Skip if already processed
    if [ -d "$OUTPUT_DIR/sparse/0" ]; then
        echo "⊘ SKIPPED (already processed)" | tee -a "$LOG_FILE"
        echo "$REL_PATH: SKIPPED" >> "$PROGRESS_FILE"
        SKIPPED=$((SKIPPED + 1))
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/sparse"

    NUM_IMAGES=$(ls "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ "$NUM_IMAGES" -eq 0 ]; then
        NUM_IMAGES=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
    fi

    echo "Images: $NUM_IMAGES" | tee -a "$LOG_FILE"

    # Feature Extraction - VERY AGGRESSIVE for low-texture medical imaging
    echo "[1/4] Extracting features (aggressive)..." | tee -a "$LOG_FILE"
    colmap feature_extractor \
        --database_path "$OUTPUT_DIR/database.db" \
        --image_path "$INPUT_DIR" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV \
        --SiftExtraction.use_gpu 0 \
        --SiftExtraction.max_image_size 3200 \
        --SiftExtraction.max_num_features 32768 \
        --SiftExtraction.peak_threshold 0.002 \
        --SiftExtraction.edge_threshold 3 \
        --SiftExtraction.first_octave -1 \
        --SiftExtraction.num_octaves 5 \
        >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "✗ FAILED at feature extraction" | tee -a "$LOG_FILE"
        echo "$REL_PATH: FAILED (feature extraction)" >> "$ERROR_FILE"
        FAILED=$((FAILED + 1))
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # EXHAUSTIVE Matching - try to match EVERY frame with EVERY other frame
    echo "[2/4] Matching features (EXHAUSTIVE - may take longer)..." | tee -a "$LOG_FILE"
    colmap exhaustive_matcher \
        --database_path "$OUTPUT_DIR/database.db" \
        --SiftMatching.use_gpu 0 \
        --SiftMatching.guided_matching 1 \
        --SiftMatching.max_ratio 0.95 \
        --SiftMatching.max_distance 0.95 \
        --SiftMatching.max_error 8.0 \
        --SiftMatching.min_num_inliers 8 \
        --SiftMatching.min_inlier_ratio 0.1 \
        >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "✗ FAILED at feature matching" | tee -a "$LOG_FILE"
        echo "$REL_PATH: FAILED (feature matching)" >> "$ERROR_FILE"
        FAILED=$((FAILED + 1))
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # Sparse Reconstruction - VERY RELAXED parameters
    echo "[3/4] Running SfM (relaxed parameters)..." | tee -a "$LOG_FILE"
    colmap mapper \
        --database_path "$OUTPUT_DIR/database.db" \
        --image_path "$INPUT_DIR" \
        --output_path "$OUTPUT_DIR/sparse" \
        --Mapper.ba_global_function_tolerance 0.000001 \
        --Mapper.init_min_tri_angle 1.5 \
        --Mapper.abs_pose_min_num_inliers 8 \
        --Mapper.abs_pose_min_inlier_ratio 0.1 \
        --Mapper.filter_min_tri_angle 1.0 \
        --Mapper.min_num_matches 8 \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        >> "$LOG_FILE" 2>&1

    if [ $? -ne 0 ] || [ ! -d "$OUTPUT_DIR/sparse/0" ]; then
        echo "✗ FAILED at reconstruction" | tee -a "$LOG_FILE"
        echo "$REL_PATH: FAILED (reconstruction)" >> "$ERROR_FILE"
        FAILED=$((FAILED + 1))
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # Export formats
    echo "[4/4] Exporting..." | tee -a "$LOG_FILE"
    colmap model_converter \
        --input_path "$OUTPUT_DIR/sparse/0" \
        --output_path "$OUTPUT_DIR/sparse/0" \
        --output_type PLY \
        >> "$LOG_FILE" 2>&1

    colmap model_converter \
        --input_path "$OUTPUT_DIR/sparse/0" \
        --output_path "$OUTPUT_DIR/sparse/0" \
        --output_type TXT \
        >> "$LOG_FILE" 2>&1

    # Get statistics
    NUM_REGISTERED=$(($(grep -v "^#" "$OUTPUT_DIR/sparse/0/images.txt" 2>/dev/null | wc -l) / 2))
    NUM_POINTS=$(grep -c '^[0-9]' "$OUTPUT_DIR/sparse/0/points3D.txt" 2>/dev/null || echo '0')

    echo "✓ SUCCESS - Registered: $NUM_REGISTERED/$NUM_IMAGES, Points: $NUM_POINTS" | tee -a "$LOG_FILE"
    echo "$REL_PATH: SUCCESS ($NUM_REGISTERED/$NUM_IMAGES images, $NUM_POINTS points)" >> "$PROGRESS_FILE"
    SUCCESS=$((SUCCESS + 1))
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "BATCH PROCESSING COMPLETE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Total directories: $TOTAL" | tee -a "$LOG_FILE"
echo "  ✓ Success: $SUCCESS" | tee -a "$LOG_FILE"
echo "  ✗ Failed: $FAILED" | tee -a "$LOG_FILE"
echo "  ⊘ Skipped: $SKIPPED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $FAILED -gt 0 ]; then
    echo "Check $ERROR_FILE for failed datasets" | tee -a "$LOG_FILE"
fi
