# RPM-Net Radar Testing Tools

This directory contains tools for testing RPM-Net on sparse radar data from Oculii sensors.

## Overview

The testing workflow consists of several stages:

1. **Data Conversion**: Convert PCD radar frames to RPM-Net compatible HDF5 format
2. **Validation**: Verify converted data quality and characteristics
3. **Single-Pair Inference**: Test registration on individual frame pairs
4. **Sequence Processing**: Process entire sequences for odometry estimation
5. **Analysis**: Analyze results and compare with ground truth

## Scripts

### 1. `convert_radar_single_frames.py`

Converts PCD radar frames to HDF5 format with normal estimation.

```bash
python convert_radar_single_frames.py
```

This script:
- Loads PCD files from `eagleg7/enhanced/`
- Estimates normals using KNN (k=20, configurable)
- Downsamples to 1024 points per frame
- Saves to `radar_single_frames/radar_single_frames_test0.h5`
- Logs normal estimation quality to `normals_log.txt`

**Key parameters to adjust:**
- `frame_indices`: Which frames to convert (default: [0, 10, 20])
- `downsample_points`: Points per frame (default: 1024)
- `k_normal`: Neighbors for normal estimation (default: 20, increase for sparse data)

### 2. `validate_converted_data.py`

Validates HDF5 structure and analyzes data quality.

```bash
# Basic validation
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5

# Visualize a frame
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5 --visualize 0

# Check overlap between frames
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5 --check_overlap 0 1
```

This script provides:
- HDF5 structure validation
- Point cloud statistics (scale, density, bounding box)
- Normal quality analysis (% random normals)
- Suggested feature radius for RPM-Net
- Frame pair overlap estimation
- Visualization with normals

### 3. `infer_single_pair.py`

Performs registration on a single frame pair.

```bash
# Basic usage with auto radius
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius \
  --neighbors 30 \
  --save_vis

# Manual radius setting
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 10 \
  --radius 3.0 \
  --neighbors 40 \
  --num_iter 10
```

**Important parameters:**
- `--auto_radius`: Automatically estimate feature radius (recommended for first test)
- `--radius`: Manual radius setting (2-5m for radar, vs 0.3m for ModelNet)
- `--neighbors`: Number of neighbors (20-50 for sparse radar)
- `--num_iter`: Registration iterations (5-10)
- `--save_vis`: Save before/after visualizations as PLY files

**Output files:**
- `pair_before.ply`: Visualization before alignment (red=src, blue=ref)
- `pair_after.ply`: Visualization after alignment
- `pred_transforms.npy`: All transforms from iterations
- `T_src{X}_ref{Y}.txt`: Final transform matrix

### 4. `process_sequence.py`

Processes multiple frame pairs for trajectory estimation.

```bash
# Process all consecutive pairs
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory

# Process with stride (every 5th frame)
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --stride 5 \
  --visualize_trajectory

# Process specific frame range
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --start_frame 0 \
  --end_frame 50
```

**Output files:**
- `sequence_results.json`: Detailed results for each pair
- `trajectory.npy`: Accumulated 3D trajectory positions
- `pairwise_transforms.npy`: All pairwise transforms
- `trajectory_visualization.png`: 3D trajectory plots (if --visualize_trajectory)

### 5. `analyze_results.py`

Analyzes sequence results and compares with ground truth.

```bash
# Basic analysis
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --output sequence_results/analysis/

# With ground truth comparison
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --ground_truth ground_truth_poses.txt \
  --output sequence_results/analysis/
```

**Output files:**
- `summary_report.txt`: Text summary of all statistics
- `rotation_translation_plot.png`: Rotation/translation over time
- `inference_time_plot.png`: Inference time statistics
- `error_comparison.png`: Error vs ground truth (if available)

### 6. `debug_helpers.py`

Utilities for debugging RPM-Net internals.

```python
# In Python debugger or notebook
from debug_helpers import *

# Print debugging guide
print_debug_breakpoint_guide()

# Inspect data
inspect_data_batch(data)

# Inspect features
inspect_features(feat_src, feat_ref)

# Inspect correspondence matrix
inspect_sinkhorn_output(perm_matrix)

# Inspect computed transform
inspect_transform(transform)
```

## Typical Workflow

### Step 1: Convert Data

```bash
cd snail_test
python convert_radar_single_frames.py
```

Check `radar_single_frames/normals_log.txt` to see normal quality.

### Step 2: Validate Conversion

```bash
python validate_converted_data.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --visualize 0 \
  --check_overlap 0 1
```

Note the suggested feature radius from the output.

### Step 3: Test Single Pair

```bash
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius \
  --neighbors 30 \
  --save_vis
```

Open `pair_before.ply` and `pair_after.ply` in CloudCompare to check alignment quality.

### Step 4: Process Sequence

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

### Step 5: Analyze Results

```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --output sequence_results/analysis/
```

## Troubleshooting

### Problem: Registration fails (large errors or no convergence)

**Possible causes:**
1. Feature radius too small → Increase `--radius` or use `--auto_radius`
2. Too few neighbors → Increase `--neighbors` to 40-50
3. Poor normal quality (>50% random) → Increase `k_normal` in conversion
4. Insufficient overlap → Choose closer frame pairs

### Problem: Normals are mostly random (>50%)

**Solutions:**
1. Increase `k_normal` in `convert_radar_single_frames.py` (try 30-50)
2. Accumulate multiple consecutive frames to increase point density
3. Use different radar processing (enhanced vs pcl vs trk)

### Problem: Registration works but trajectory drifts

**This is expected for sequential odometry:**
1. Errors accumulate over time
2. Consider loop closure techniques
3. Compare drift rate with ground truth
4. Use larger stride to reduce accumulated error

## Parameter Tuning Guide

### Feature Radius

- **ModelNet (dense)**: 0.3m
- **Radar (sparse)**: 2-5m
- **Auto-estimation**: Usually reliable, use as starting point
- **Too small**: No features found, registration fails
- **Too large**: Features not discriminative, poor matches

### Number of Neighbors

- **ModelNet**: 64
- **Radar**: 20-50
- **Sparse data**: Increase to ensure enough neighbors
- **Dense data**: Can decrease for speed

### Registration Iterations

- **Default**: 5
- **Difficult cases**: 10
- **Diminishing returns**: Usually converges by iteration 5-10
- **Check convergence**: Look at transforms in `debug_helpers`

## Data Requirements

For good registration results:
- **Overlap**: >50% between consecutive frames
- **Point density**: >100 points per frame (after downsampling to 1024)
- **Normal quality**: <30% random normals
- **Motion**: Rotation <30° and translation <5m between pairs

## Expected Performance

### Good Results
- Rotation error: <5°
- Translation error: <0.5m
- Inference time: 50-200ms per pair

### Acceptable Results
- Rotation error: <15°
- Translation error: <2m
- May require parameter tuning

### Poor Results
- Rotation error: >30°
- Translation error: >5m
- Indicates fundamental issue (overlap, normals, parameters)

