# Quick Start Guide: Testing RPM-Net with Radar Data

This guide helps you get started quickly with testing RPM-Net on your Oculii radar data.

## Prerequisites Check

```bash
# Verify you have the required packages
pip install numpy h5py open3d scikit-learn matplotlib pandas torch tqdm

# Verify your data structure
ls eagleg7/enhanced/  # Should show .pcd files
```

## 5-Minute Quick Test

### 1. Convert 3 test frames (30 seconds)

```bash
cd snail_test
python convert_radar_single_frames.py
```

**Expected output:**
- `radar_single_frames/radar_single_frames_test0.h5`
- `radar_single_frames/normals_log.txt`
- Should complete in ~30 seconds

**Check the log:**
```bash
cat radar_single_frames/normals_log.txt
```

Look for the random normal percentage. Target: <30%

### 2. Validate the converted data (10 seconds)

```bash
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5
```

**What to look for:**
- ✓ Structure validation passed
- Note the "Suggested feature radius" (important for next step!)
- Check that random normals are <30% per frame

### 3. Test a single frame pair (5 seconds)

```bash
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume YOUR_CHECKPOINT_PATH.pth \
  --src 0 --ref 1 \
  --auto_radius \
  --neighbors 30 \
  --save_vis
```

**Replace `YOUR_CHECKPOINT_PATH.pth`** with the actual path to your checkpoint, e.g.:
- `D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth`

**Expected output:**
```
Auto-estimated feature radius: ~2-5m
Rotation = X.XX deg, translation = X.XXX m
Saved PLY to .../radar_single_frames (pair_before.ply, pair_after.ply)
```

### 4. Visualize the result (manual)

Open the PLY files in any point cloud viewer:
- **CloudCompare** (recommended): File → Open → select both PLY files
- **Open3D viewer**: They're already colored (red=source, blue=reference)

**What to check:**
- `pair_before.ply`: Should show two separate point clouds (red and blue)
- `pair_after.ply`: Red cloud should align with blue cloud

## Understanding Your First Results

### Good Registration Signs
✓ Rotation: 0-5°  
✓ Translation: 0.1-1m (depends on vehicle motion)  
✓ Visual alignment in `pair_after.ply`

### Warning Signs
⚠️ Rotation: >30°  
⚠️ Translation: >5m  
⚠️ Clouds don't align visually

### If Results Look Bad

**Try these fixes in order:**

1. **Increase feature radius**
   ```bash
   python infer_single_pair.py --h5 ... --resume ... --radius 5.0 --neighbors 40
   ```

2. **Increase neighbors**
   ```bash
   python infer_single_pair.py --h5 ... --resume ... --auto_radius --neighbors 50
   ```

3. **More iterations**
   ```bash
   python infer_single_pair.py --h5 ... --resume ... --auto_radius --num_iter 10
   ```

4. **Re-convert with better normals**
   Edit `convert_radar_single_frames.py` line 246:
   ```python
   k_normal=30,  # Increased from 20
   ```
   Then re-run conversion.

## Next Steps

Once the single-pair test works:

### Process a Short Sequence (1 minute)

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume YOUR_CHECKPOINT_PATH.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

This processes all frame pairs (0→1, 1→2) and estimates trajectory.

### Analyze Results (5 seconds)

```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --output sequence_results/analysis/
```

Check `sequence_results/analysis/summary_report.txt` for statistics.

## Full Workflow for Real Testing

Once quick tests pass, process more frames:

### 1. Convert more frames

Edit `convert_radar_single_frames.py` line 244:
```python
frame_indices=list(range(0, 100, 2)),  # Every 2nd frame, 0-100
```

Re-run:
```bash
python convert_radar_single_frames.py
```

### 2. Process full sequence

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume YOUR_CHECKPOINT_PATH.pth \
  --output_dir full_sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory \
  --verbose
```

### 3. Analyze

```bash
python analyze_results.py \
  --results full_sequence_results/sequence_results.json \
  --output full_sequence_results/analysis/
```

## Troubleshooting

### Error: "File not found"
- Check paths are correct
- Make sure you're in the `snail_test/` directory

### Error: "CUDA out of memory"
- Add `--gpu -1` to use CPU instead
- Or reduce batch processing

### Error: "No module named ..."
- Install missing package: `pip install <module_name>`
- Or run: `pip install -r ../requirements.txt` from RPMNet root

### Poor results even after tuning
- Your radar data may be too sparse
- Try accumulating multiple consecutive frames
- Check the data directory (try `enhanced/` vs `pcl/` vs `trk/`)

### Normals >50% random
- Increase `k_normal` to 40-50
- Or accumulate 2-3 frames before downsampling
- This is a fundamental data sparsity issue

## Performance Expectations

### Radar vs ModelNet

| Metric | ModelNet (dense) | Radar (sparse) |
|--------|------------------|----------------|
| Rotation error | <1° | <5° (good), <15° (acceptable) |
| Translation error | <0.1m | <0.5m (good), <2m (acceptable) |
| Feature radius | 0.3m | 2-5m |
| Neighbors | 64 | 20-50 |
| Random normals | <5% | <30% target |

### Why Radar is Harder

1. **Sparsity**: 1024 points spread over 50m vs 1m object
2. **Normals**: Estimated from sparse neighbors vs known
3. **Partial data**: Moving objects, occlusion, varying returns
4. **Scale**: Larger scenes need larger feature neighborhoods

## Getting Help

If you're stuck:

1. Check `radar_single_frames/normals_log.txt` for data quality
2. Run validation script for statistics
3. Try the debug helpers (see `debug_helpers.py`)
4. Compare your results with expected ranges above

## Summary Checklist

- [ ] Converted 3 test frames successfully
- [ ] Validation shows <30% random normals
- [ ] Single pair registration completes without errors
- [ ] Visual alignment in pair_after.ply looks reasonable
- [ ] Rotation and translation are in expected ranges
- [ ] Ready to process full sequence

If all boxes are checked, you're ready for production testing! 🚀

