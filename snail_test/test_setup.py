"""Test script to verify your setup is ready for RPM-Net radar testing

This script checks that all dependencies, data, and models are accessible.

Usage:
    python test_setup.py --checkpoint /path/to/model.pth
"""
import argparse
import sys
from pathlib import Path

# Test imports
print("="*60)
print("Testing Python Package Imports")
print("="*60)

required_packages = [
    ('numpy', 'np'),
    ('h5py', 'h5py'),
    ('open3d', 'o3d'),
    ('sklearn', 'sklearn'),
    ('torch', 'torch'),
    ('matplotlib', 'plt'),
    ('pandas', 'pd'),
    ('tqdm', 'tqdm'),
]

missing_packages = []
for package, import_name in required_packages:
    try:
        if import_name == 'plt':
            exec(f"import matplotlib.pyplot as {import_name}")
        else:
            exec(f"import {import_name}")
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n✓ All required packages installed")

# Check project structure
print("\n" + "="*60)
print("Testing Project Structure")
print("="*60)

script_dir = Path(__file__).parent
project_root = script_dir.parent

required_paths = [
    ('src/', 'RPM-Net source code directory'),
    ('src/models/rpmnet.py', 'RPMNet model file'),
    ('src/eval.py', 'Evaluation script'),
    ('snail_test/eagleg7/', 'Radar data directory'),
]

missing_paths = []
for rel_path, description in required_paths:
    full_path = project_root / rel_path
    if full_path.exists():
        print(f"✓ {description}: {full_path}")
    else:
        print(f"✗ {description}: NOT FOUND at {full_path}")
        missing_paths.append(rel_path)

if missing_paths:
    print(f"\n⚠️  Some paths are missing, but this may be OK if you haven't set up data yet")
else:
    print("\n✓ All expected paths found")

# Check radar data
print("\n" + "="*60)
print("Testing Radar Data")
print("="*60)

radar_dirs = [
    script_dir / 'eagleg7' / 'enhanced',
    script_dir / 'eagleg7' / 'pcl',
    script_dir / 'eagleg7' / 'trk',
]

pcd_counts = {}
for radar_dir in radar_dirs:
    if radar_dir.exists():
        pcd_files = list(radar_dir.glob('*.pcd'))
        pcd_counts[radar_dir.name] = len(pcd_files)
        print(f"✓ {radar_dir.name}/: {len(pcd_files)} PCD files")
    else:
        print(f"✗ {radar_dir.name}/: NOT FOUND")
        pcd_counts[radar_dir.name] = 0

if sum(pcd_counts.values()) == 0:
    print("\n⚠️  No PCD files found. Make sure radar data is in snail_test/eagleg7/")
else:
    print(f"\n✓ Total PCD files: {sum(pcd_counts.values())}")

# Check if model checkpoint is provided
print("\n" + "="*60)
print("Testing RPM-Net Model")
print("="*60)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, 
                   help='Path to RPM-Net checkpoint .pth file')
args = parser.parse_args()

if args.checkpoint is None:
    print("ℹ️  No checkpoint provided (use --checkpoint to test model loading)")
    print("   You'll need a checkpoint to run inference")
else:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
    else:
        print(f"✓ Checkpoint found: {checkpoint_path}")
        
        # Try to load the model
        try:
            import torch
            from types import SimpleNamespace
            
            # Add src to path
            src_dir = project_root / 'src'
            sys.path.insert(0, str(src_dir))
            
            import models.rpmnet as rpmnet_mod
            
            # Create dummy args
            rpm_args = SimpleNamespace(
                method='rpmnet',
                features=['ppf', 'dxyz', 'xyz'],
                feat_dim=96,
                radius=2.0,
                num_neighbors=30,
                num_reg_iter=5,
                add_slack=True,
                no_slack=False,
                num_sk_iter=5
            )
            
            # Build model
            model = rpmnet_mod.get_model(rpm_args)
            
            # Load weights
            state = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
            
            print("✓ Model loaded successfully")
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model has {n_params:,} parameters")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")

# Check converted data if exists
print("\n" + "="*60)
print("Testing Converted Data")
print("="*60)

h5_path = script_dir / 'radar_single_frames' / 'radar_single_frames_test0.h5'
if h5_path.exists():
    print(f"✓ Found converted data: {h5_path}")
    
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            print(f"  Keys: {list(f.keys())}")
            if 'data' in f:
                print(f"  Data shape: {f['data'].shape}")
            if 'normal' in f:
                print(f"  Normal shape: {f['normal'].shape}")
        print("✓ HDF5 file is valid")
    except Exception as e:
        print(f"✗ Failed to read HDF5: {e}")
else:
    print(f"ℹ️  No converted data found at {h5_path}")
    print("   Run convert_radar_single_frames.py to create it")

# Check helper scripts
print("\n" + "="*60)
print("Testing Helper Scripts")
print("="*60)

helper_scripts = [
    'convert_radar_single_frames.py',
    'validate_converted_data.py',
    'infer_single_pair.py',
    'process_sequence.py',
    'analyze_results.py',
    'debug_helpers.py',
]

for script in helper_scripts:
    script_path = script_dir / script
    if script_path.exists():
        print(f"✓ {script}")
    else:
        print(f"✗ {script} - MISSING")

# Final summary
print("\n" + "="*60)
print("Setup Summary")
print("="*60)

issues = []
if missing_packages:
    issues.append("Missing Python packages")
if sum(pcd_counts.values()) == 0:
    issues.append("No radar data found")
if args.checkpoint and not Path(args.checkpoint).exists():
    issues.append("Checkpoint not found")

if issues:
    print("\n⚠️  Setup has issues:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nResolve these issues before proceeding.")
else:
    print("\n✓ Setup looks good!")
    print("\nNext steps:")
    print("1. Run: python convert_radar_single_frames.py")
    print("2. Run: python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5")
    print("3. Run: python infer_single_pair.py --h5 ... --resume YOUR_CHECKPOINT.pth --src 0 --ref 1 --auto_radius --save_vis")
    print("\nSee QUICKSTART.md for detailed instructions!")

print("\n" + "="*60)

