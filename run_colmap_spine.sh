#!/bin/bash
# COLMAP pipeline for spine endoscopy SfM preprocessing
# This generates the required data for depth estimation training

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

# Where your 640×360 spine endoscopy frames are stored
INPUT_IMAGES="/path/to/your/spine/frames"

# Where to save COLMAP outputs
WORKSPACE="/path/to/colmap/workspace"

# Create workspace directories
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/sparse"
mkdir -p "$WORKSPACE/dense"

# ============================================
# STEP 1: Feature Extraction
# ============================================
echo "Step 1/4: Extracting features from images..."

colmap feature_extractor \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$INPUT_IMAGES" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 8192

# Explanation:
# - single_camera 1: All frames from same camera (endoscope)
# - camera_model OPENCV: Handles lens distortion
# - max_num_features 8192: More features = better reconstruction

# ============================================
# STEP 2: Feature Matching
# ============================================
echo "Step 2/4: Matching features across frames..."

# For sequential video frames (RECOMMENDED for endoscopy):
colmap sequential_matcher \
    --database_path "$WORKSPACE/database.db" \
    --SiftMatching.guided_matching 1 \
    --SequentialMatching.overlap 10 \
    --SequentialMatching.loop_detection 1

# Alternative for unordered images:
# colmap exhaustive_matcher \
#     --database_path "$WORKSPACE/database.db"

# Explanation:
# - sequential_matcher: Faster, assumes frames are in order
# - overlap 10: Match each frame with next 10 frames
# - loop_detection 1: Find when camera returns to same place

# ============================================
# STEP 3: Sparse Reconstruction (SfM)
# ============================================
echo "Step 3/4: Running Structure from Motion..."

colmap mapper \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$INPUT_IMAGES" \
    --output_path "$WORKSPACE/sparse" \
    --Mapper.ba_global_function_tolerance 0.000001

# This creates:
# - cameras.bin: Camera parameters
# - images.bin: Camera poses
# - points3D.bin: Sparse 3D point cloud

# ============================================
# STEP 4: Export to Required Format
# ============================================
echo "Step 4/4: Exporting results..."

# Export sparse point cloud to PLY format
colmap model_converter \
    --input_path "$WORKSPACE/sparse/0" \
    --output_path "$WORKSPACE/sparse/0" \
    --output_type PLY

# Export camera poses to TXT format for easier parsing
colmap model_converter \
    --input_path "$WORKSPACE/sparse/0" \
    --output_path "$WORKSPACE/sparse/0" \
    --output_type TXT

echo "================================"
echo "COLMAP reconstruction complete!"
echo "================================"
echo ""
echo "Outputs saved to: $WORKSPACE/sparse/0/"
echo ""
echo "Generated files:"
echo "  - points3D.ply      → Rename to structure.ply"
echo "  - cameras.txt       → Extract intrinsics"
echo "  - images.txt        → Extract poses for motion.yaml"
echo ""
echo "Next steps:"
echo "1. Convert COLMAP output to required format (see convert_colmap.py)"
echo "2. Generate undistorted_mask.bmp from valid image regions"
echo "3. Create view_indexes_per_point file"
echo "4. Organize into format: bag_X/sequence_Y/"
echo ""
echo "TIP: Check reconstruction quality:"
echo "  colmap gui --import_path $WORKSPACE/sparse/0"
