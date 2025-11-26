#!/bin/bash
# COLMAP pipeline for spine endoscopy (HEADLESS MODE for servers)

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

# Where your spine endoscopy frames are stored
INPUT_IMAGES="/home/test1/workspace/test-data"

# Where to save COLMAP outputs
WORKSPACE="/home/test1/workspace/test-colmap-output"

# ============================================
# IMPORTANT: Run COLMAP in headless mode
# ============================================
export QT_QPA_PLATFORM=offscreen

# Create workspace directories
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/sparse"

# Check inputs
echo "=== Configuration ==="
echo "Images: $INPUT_IMAGES"
echo "Output: $WORKSPACE"
echo "Platform: $QT_QPA_PLATFORM (headless mode)"
echo ""

NUM_IMAGES=$(ls "$INPUT_IMAGES"/*.jpg 2>/dev/null | wc -l)
if [ "$NUM_IMAGES" -eq 0 ]; then
    NUM_IMAGES=$(ls "$INPUT_IMAGES"/*.png 2>/dev/null | wc -l)
fi

echo "Found $NUM_IMAGES images"
echo ""

# ============================================
# STEP 1: Feature Extraction
# ============================================
echo "Step 1/4: Extracting features from images..."

colmap feature_extractor \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$INPUT_IMAGES" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192

if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed!"
    exit 1
fi

echo "✓ Feature extraction complete"
echo ""

# ============================================
# STEP 2: Feature Matching
# ============================================
echo "Step 2/4: Matching features across frames..."

colmap sequential_matcher \
    --database_path "$WORKSPACE/database.db" \
    --SiftMatching.use_gpu 0 \
    --SiftMatching.guided_matching 1 \
    --SequentialMatching.overlap 10 \
    --SequentialMatching.loop_detection 1

if [ $? -ne 0 ]; then
    echo "ERROR: Feature matching failed!"
    exit 1
fi

echo "✓ Feature matching complete"
echo ""

# ============================================
# STEP 3: Sparse Reconstruction (SfM)
# ============================================
echo "Step 3/4: Running Structure from Motion..."

colmap mapper \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$INPUT_IMAGES" \
    --output_path "$WORKSPACE/sparse" \
    --Mapper.ba_global_function_tolerance 0.000001

if [ $? -ne 0 ]; then
    echo "ERROR: Mapping failed!"
    echo "Check if images have enough features and overlap"
    exit 1
fi

echo "✓ Mapping complete"
echo ""

# Check if reconstruction was created
if [ ! -d "$WORKSPACE/sparse/0" ]; then
    echo "ERROR: No reconstruction created!"
    echo "Possible issues:"
    echo "  - Images don't have enough texture/features"
    echo "  - Images don't overlap enough"
    echo "  - Camera moved too fast between frames"
    exit 1
fi

# ============================================
# STEP 4: Export to Required Format
# ============================================
echo "Step 4/4: Exporting results..."

# Export to PLY
colmap model_converter \
    --input_path "$WORKSPACE/sparse/0" \
    --output_path "$WORKSPACE/sparse/0" \
    --output_type PLY

# Export to TXT
colmap model_converter \
    --input_path "$WORKSPACE/sparse/0" \
    --output_path "$WORKSPACE/sparse/0" \
    --output_type TXT

echo ""
echo "================================"
echo "✓ COLMAP reconstruction complete!"
echo "================================"
echo ""
echo "Outputs saved to: $WORKSPACE/sparse/0/"
echo ""
echo "Generated files:"
ls -lh "$WORKSPACE/sparse/0/"
echo ""
echo "Statistics:"
echo "  Images: $(grep -c '^[0-9]' "$WORKSPACE/sparse/0/images.txt" 2>/dev/null || echo 'N/A')"
echo "  3D points: $(grep -c '^[0-9]' "$WORKSPACE/sparse/0/points3D.txt" 2>/dev/null || echo 'N/A')"
echo ""
echo "Next step: Convert to training format with convert_colmap_to_required_format.py"
