# RPM-Net Radar Testing - Complete Guide

This document provides a comprehensive plan for testing RPM-Net on sparse Oculii radar data.

## Table of Contents

1. [Understanding RPM-Net](#1-understanding-rpm-net)
2. [Data Conversion](#2-data-conversion)
3. [Testing Workflow](#3-testing-workflow)
4. [Parameter Tuning](#4-parameter-tuning)
5. [Evaluation](#5-evaluation)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Understanding RPM-Net

### Core Architecture (Read First)

**Critical files to understand:**

1. **`src/models/rpmnet.py`** - Main registration loop
   - Line 162-215: `RPMNet.forward()` - MOST IMPORTANT
   - Line 186-188: Feature extraction from xyz + normals
   - Line 190-191: Compute feature distance → affinity matrix
   - Line 194-196: Sinkhorn algorithm → soft correspondences
   - Line 199: Weighted SVD → rigid transform
   - Line 200: Apply transform iteratively

2. **`src/data_loader/datasets.py`** - Data loading
   - Line 208: Concatenates xyz (3D) + normals (3D) → 6D points
   - Understands HDF5 format with 'data' and 'normal' datasets

3. **`src/models/feature_nets.py`** - Feature extraction
   - `FeatExtractionEarlyFusion`: Extracts PPF, xyz, dxyz features
   - Uses radius and num_neighbors to define local neighborhoods

### Key Concepts

**Point Pair Features (PPF):**
- Geometric features computed from local point neighborhoods
- Requires normals and local structure
- More discriminative than just xyz coordinates

**Sinkhorn Algorithm:**
- Converts feature distances → soft correspondence matrix
- Each source point gets weighted matches to multiple reference points
- "Slack" column handles outliers

**Iterative Registration:**
- Refines transform over multiple iterations
- Each iteration: extract features → match → estimate transform → apply

### Debugging Strategy

**Set breakpoints at (in `src/models/rpmnet.py`):**
- Line 184: Start of iteration loop
- Line 186: Check beta, alpha parameters
- Line 188: Check extracted features
- Line 194: Check affinity matrix
- Line 196: Check correspondence matrix
- Line 199: Check computed transform

**Use debug helpers:**
```python
from snail_test.debug_helpers import *
inspect_data_batch(data)
inspect_features(feat_src, feat_ref)
inspect_sinkhorn_output(perm_matrix)
inspect_transform(transform)
```

---

## 2. Data Conversion

### Your Current Setup

**Input:** PCD files in `snail_test/eagleg7/enhanced/`  
**Output:** HDF5 file at `snail_test/radar_single_frames/radar_single_frames_test0.h5`

**Conversion script:** `snail_test/convert_radar_single_frames.py`

### Critical Parameters

```python
# In convert_radar_single_frames.py, main section:
convert_single_radar_frames(
    radar_dir="./eagleg7/enhanced/",
    output_dir="radar_single_frames/",
    frame_indices=[0, 10, 20],      # Which frames to convert
    downsample_points=1024,          # Points per frame
    k_normal=20,                     # Neighbors for normal estimation
    log_filename="normals_log.txt"
)
```

**Key parameter: `k_normal`**
- Default: 20
- Sparse data: Increase to 30-50
- Very sparse: Consider accumulating multiple frames
- Target: <30% random normals (check in log file)

### Data Quality Checklist

**Before conversion:**
- [ ] PCD files exist and contain >100 points each
- [ ] Coordinate system is in meters (not millimeters)
- [ ] Understand which data type to use: enhanced, pcl, or trk

**After conversion:**
- [ ] HDF5 file created successfully
- [ ] Check `normals_log.txt`: random normals <30%
- [ ] Run validation script (see below)

### Validation Command

```bash
cd snail_test
python validate_converted_data.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --visualize 0 \
  --check_overlap 0 1
```

**What to look for:**
- Structure validation: PASS
- Random normals: <30% per frame
- Suggested feature radius: Note this value!
- Overlap percentage: >50% for adjacent frames

---

## 3. Testing Workflow

### Step-by-Step Process

#### Step 0: Setup Check (1 minute)

```bash
cd snail_test
python test_setup.py --checkpoint /path/to/your/checkpoint.pth
```

This verifies all dependencies and data are ready.

#### Step 1: Convert Test Frames (30 seconds)

```bash
python convert_radar_single_frames.py
```

Check the log: `cat radar_single_frames/normals_log.txt`

#### Step 2: Validate Data (10 seconds)

```bash
python validate_converted_data.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5
```

**Note the suggested feature radius from output!**

#### Step 3: Single Pair Test (5 seconds)

```bash
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius \
  --neighbors 30 \
  --save_vis
```

**Expected output:**
```
Auto-estimated feature radius: 2.XXX
=== RPMNet result (src -> ref) ===
rotation = X.XX deg, translation = X.XXX m
Saved PLY to .../radar_single_frames
```

#### Step 4: Visualize Results (manual)

Open `pair_before.ply` and `pair_after.ply` in CloudCompare or similar viewer.
- Before: Two separate clouds (red and blue)
- After: Aligned (red should overlap blue)

#### Step 5: Process Sequence (1 minute)

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

**Output files:**
- `sequence_results/sequence_results.json`
- `sequence_results/trajectory.npy`
- `sequence_results/trajectory_visualization.png`

#### Step 6: Analyze Results (5 seconds)

```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --output sequence_results/analysis/
```

Check: `sequence_results/analysis/summary_report.txt`

---

## 4. Parameter Tuning

### Critical Parameters

#### 1. Feature Radius (`--radius` or `--auto_radius`)

**Purpose:** Defines neighborhood size for feature extraction

| Data Type | Recommended Range |
|-----------|-------------------|
| ModelNet (dense objects) | 0.3 m |
| Radar (sparse scenes) | 2-5 m |
| Very sparse radar | 5-10 m |

**How to choose:**
- Use `--auto_radius` first (recommended)
- If registration fails, manually increase
- Too small → no features found
- Too large → features not discriminative

**Example:**
```bash
# Auto-estimate (recommended)
--auto_radius

# Manual setting
--radius 3.5
```

#### 2. Number of Neighbors (`--neighbors`)

**Purpose:** How many nearest points to use for each feature

| Data Type | Recommended |
|-----------|-------------|
| ModelNet | 64 |
| Radar (moderate) | 30 |
| Radar (sparse) | 40-50 |

**Guidelines:**
- Increase if "not enough neighbors" errors
- Decrease for speed (if data is dense)
- Must be ≤ num_points (1024)

#### 3. Registration Iterations (`--num_iter`)

**Purpose:** How many times to refine the transform

| Scenario | Recommended |
|----------|-------------|
| Default | 5 |
| Difficult cases | 10 |
| Very difficult | 15 |

**Guidelines:**
- Usually converges by iteration 5-10
- Check convergence with debug helpers
- More iterations = more time but better results

#### 4. Normal Estimation (`k_normal` in conversion)

**Purpose:** Neighbors for PCA-based normal estimation

| Data Density | Recommended |
|--------------|-------------|
| Moderate | 20 |
| Sparse | 30-40 |
| Very sparse | 50+ or accumulate frames |

**Target:** <30% random normals

### Parameter Tuning Flowchart

```
Start with defaults:
  --auto_radius --neighbors 30 --num_iter 5

Registration fails?
  ↓
Check overlap > 50%? (use validate script)
  No → Choose closer frames
  Yes ↓
  
Check random normals < 30%? (in log file)
  No → Increase k_normal in conversion, re-convert
  Yes ↓
  
Try: --radius 5.0 --neighbors 40
  Still fails? ↓
  
Try: --radius 7.0 --neighbors 50 --num_iter 10
  Still fails? ↓
  
Data may be too sparse:
  - Accumulate multiple consecutive frames
  - Try different data type (enhanced vs pcl vs trk)
```

---

## 5. Evaluation

### Performance Metrics

#### Without Ground Truth

**Visual Inspection:**
1. Load `pair_after.ply` in viewer
2. Check if red cloud aligns with blue cloud
3. Subjective score: 1 (bad) to 5 (perfect)

**Consistency Check:**
1. Process full loop (e.g., frames 0→10→20→0)
2. Chain transforms should return near identity
3. Large loop closure error = accumulated drift

**Trajectory Smoothness:**
1. Visualize trajectory (from `process_sequence.py`)
2. Should be smooth, no sudden jumps
3. Jumps indicate registration failures

#### With Ground Truth

**Metrics (use `analyze_results.py`):**

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Rotation error | <5° | 5-15° | >15° |
| Translation error | <0.5m | 0.5-2m | >2m |
| Chamfer distance | <0.01 | 0.01-0.1 | >0.1 |

**Comparison script:**
```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --ground_truth ground_truth_poses.txt \
  --output sequence_results/analysis/
```

### Expected Performance: Radar vs ModelNet

| Aspect | ModelNet | Radar |
|--------|----------|-------|
| Point density | ~1 pt/0.001 m³ | ~1 pt/0.1 m³ |
| Scene scale | 1m objects | 50m scenes |
| Normal quality | Known/perfect | Estimated |
| Typical rot error | <1° | <5° (good) |
| Typical trans error | <0.05m | <0.5m (good) |
| Feature radius | 0.3m | 2-5m |

**Why radar is harder:**
1. Sparse data → less reliable features
2. Large scale → larger search radius needed
3. Estimated normals → some random/wrong
4. Partial observations → less overlap

---

## 6. Troubleshooting

### Common Issues

#### Issue 1: "CUDA out of memory"

**Solutions:**
```bash
# Use CPU instead
--gpu -1

# Or use smaller batch (in process_sequence.py, this shouldn't happen)
```

#### Issue 2: Random normals >50%

**Cause:** Data too sparse for reliable normal estimation

**Solutions:**
1. Increase `k_normal` in conversion (try 30, 40, 50)
2. Accumulate 2-3 consecutive frames before downsampling
3. Try different data source (enhanced vs pcl vs trk)

**Example fix:**
```python
# In convert_radar_single_frames.py
def accumulate_frames(pcd_files, indices, accumulate=3):
    """Accumulate multiple frames to increase density"""
    all_points = []
    for i in indices:
        for j in range(accumulate):
            if i+j < len(pcd_files):
                pcd = o3d.io.read_point_cloud(str(pcd_files[i+j]))
                all_points.append(np.asarray(pcd.points))
    return np.vstack(all_points)
```

#### Issue 3: Registration completely fails (large errors)

**Debug checklist:**
1. Check overlap: `validate_converted_data.py --check_overlap X Y`
   - If <30%, choose closer frames
2. Check feature radius: Try larger radius
3. Check normal quality: Should be <30% random
4. Visualize input: `--visualize X` to check data quality

**Systematic debugging:**
```bash
# Step 1: Verify data quality
python validate_converted_data.py --h5 ... --visualize 0 --check_overlap 0 1

# Step 2: Try with large radius and more neighbors
python infer_single_pair.py --h5 ... --resume ... \
  --radius 5.0 --neighbors 50 --num_iter 10 --save_vis

# Step 3: Check if problem is fundamental or parameter issue
# If still fails with large radius + many neighbors → data issue
```

#### Issue 4: Works for frame 0→1 but fails for 0→10

**Cause:** Too large motion between frames

**Solutions:**
1. Use smaller stride (process more intermediate frames)
2. Use `process_sequence.py` to chain transforms
3. Increase `--num_iter` for difficult pairs

#### Issue 5: Trajectory drifts over time

**This is expected!** Sequential odometry accumulates error.

**Mitigations:**
1. Use ground truth for comparison
2. Implement loop closure
3. Fuse with other sensors (IMU, wheel odometry)
4. Process with larger stride (accumulate fewer errors)

### Data Quality Issues

#### Symptoms of poor data quality:
- Random normals >50%
- Very sparse point clouds (<100 points after filtering)
- Large bounding box (>100m) with few points
- No structure visible in visualization

#### Solutions:
1. **Try different data type:**
   - `enhanced/`: Usually best
   - `pcl/`: Raw point cloud
   - `trk/`: Tracked targets (may be too sparse)

2. **Accumulate frames:**
   - Combine 2-3 consecutive frames
   - Increases density, improves normals
   - Trade-off: more motion blur

3. **Adjust radar processing:**
   - If you have access to raw radar data
   - Tune detection thresholds
   - Apply better filtering

### When to Give Up

Some data may be fundamentally unsuitable:
- <50 points per frame after processing
- Random normals >70% even with high k_normal
- No overlap between consecutive frames
- Purely dynamic scenes (no static structure)

In these cases, consider:
- Using learning-based odometry specifically for radar
- Fusing with other sensors
- Improving radar processing pipeline

---

## Quick Reference

### Command Cheat Sheet

```bash
# Convert data
python convert_radar_single_frames.py

# Validate
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5

# Single pair test
python infer_single_pair.py --h5 FILE.h5 --resume MODEL.pth --src 0 --ref 1 --auto_radius --save_vis

# Process sequence
python process_sequence.py --h5 FILE.h5 --resume MODEL.pth --output_dir results/ --auto_radius --visualize_trajectory

# Analyze
python analyze_results.py --results results/sequence_results.json --output results/analysis/
```

### File Structure

```
RPMNet/
├── src/                          # RPM-Net source code
│   ├── models/rpmnet.py         # Main model (READ THIS!)
│   ├── data_loader/datasets.py  # Data loading
│   └── eval.py                  # Evaluation
├── snail_test/                  # Your testing scripts
│   ├── eagleg7/enhanced/        # Input: PCD files
│   ├── convert_radar_single_frames.py
│   ├── validate_converted_data.py
│   ├── infer_single_pair.py
│   ├── process_sequence.py
│   ├── analyze_results.py
│   ├── debug_helpers.py
│   ├── test_setup.py
│   ├── radar_single_frames/     # Output: HDF5 + metadata
│   │   ├── radar_single_frames_test0.h5
│   │   ├── normals_log.txt
│   │   └── normal_is_random.npy
│   └── sequence_results/        # Results from processing
│       ├── sequence_results.json
│       ├── trajectory.npy
│       └── analysis/
└── TESTING_GUIDE.md             # This file
```

---

## Summary

Your goal: Test RPM-Net on sparse Oculii radar for odometry estimation.

Your code foundation is solid:
- ✓ Conversion script ready
- ✓ Single-pair inference ready
- ✓ Helper scripts created

Main challenges:
1. **Data sparsity** → Tune radius/neighbors
2. **Normal quality** → Tune k_normal
3. **Parameter optimization** → Iterative tuning

Success criteria:
- Single pair: <5° rotation, <1m translation
- Sequence: Smooth trajectory, <15° avg rotation error

**Next action:** Run test_setup.py, then follow QUICKSTART.md!

