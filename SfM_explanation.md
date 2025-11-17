# Structure from Motion (SfM) Explained

## Input: Just Images
You have: `image_001.jpg, image_002.jpg, ..., image_N.jpg`

## SfM Process:

### Step 1: Feature Detection
- Finds distinctive points in each image (corners, edges, textures)
- Example: 2000-5000 feature points per frame

### Step 2: Feature Matching
- Matches the same physical point across multiple images
- Example: Point on spine surface visible in frames 10, 11, 12, 13

### Step 3: Camera Pose Estimation
- Figures out WHERE the camera was when each photo was taken
- Computes: position (x,y,z) + orientation (rotation)

### Step 4: 3D Triangulation
- Uses matched points + camera poses to compute 3D coordinates
- Creates a SPARSE 3D point cloud (not dense!)

## Outputs Generated:

1. **structure.ply**
   - Sparse 3D point cloud (e.g., 10,000 points)
   - Each point: (x, y, z, r, g, b)

2. **motion.yaml** or camera poses
   - For each image: 4×4 transformation matrix
   - Tells where camera was located + oriented

3. **camera_intrinsics_per_view**
   - Camera focal length, principal point
   - Calibration data

4. **view_indexes_per_point**
   - Which images see each 3D point
   - Used for consistency checking

## Why This Depth Estimation Method Needs SfM:

### Self-Supervised Learning Requires Supervision Signals:

**Without SfM:**
- No ground truth depth labels
- No way to know if predicted depth is correct
- Can't train!

**With SfM:**
1. **Sparse depth supervision**: Use sparse 3D points as weak depth labels
2. **Geometric constraints**: Camera poses enable consistency checks
3. **Scale recovery**: Sparse points provide metric scale
4. **Flow supervision**: Camera motion + depth → optical flow to verify

## Analogy:
Think of training a model to predict depths like learning to draw 3D:

- **Without SfM**: "Draw a 3D scene" with NO reference → impossible
- **With SfM**: "Here are some 3D points and camera angles, now predict the full 3D" → learnable!

SfM provides the "scaffolding" that lets the neural network learn depth in a self-supervised way.
