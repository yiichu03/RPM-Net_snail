"""Process a sequence of radar frames for odometry estimation

This script extends infer_single_pair.py to process consecutive frame pairs
and accumulate transforms to estimate a trajectory.

Usage:
    # Process all consecutive pairs in the H5 file
    python process_sequence.py \
        --h5 radar_single_frames/radar_single_frames_test0.h5 \
        --resume path/to/checkpoint.pth \
        --output_dir sequence_results/ \
        --auto_radius --neighbors 30

    # Process with stride (e.g., every 5th frame)
    python process_sequence.py \
        --h5 radar_single_frames/radar_single_frames_test0.h5 \
        --resume path/to/checkpoint.pth \
        --stride 5 \
        --output_dir sequence_results/
"""
from pathlib import Path
import sys
import argparse
import os
from types import SimpleNamespace
import json
import time

import h5py
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import models.rpmnet as rpmnet_mod


def to_tensor(x, device):
    return torch.from_numpy(x).float().to(device)


def apply_se3(xyz, T):
    """Apply SE3 transform to points"""
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    if T.shape == (3, 4):
        R, t = T[:, :3], T[:, 3]
    elif T.shape == (4, 4):
        R, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError("T must be (3,4) or (4,4)")
    return xyz @ R.T + t


def auto_radius(points_src, points_ref, k=16, pct=80):
    """Estimate feature radius from kNN distances"""
    import sklearn.neighbors as skn
    pts = np.vstack([points_src, points_ref])
    nbrs = skn.NearestNeighbors(n_neighbors=min(k+1, len(pts))).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    base = np.median(dists[:, 1:])
    return float(np.percentile(dists[:, 1:], pct)) if np.isfinite(base) else 1.0


def robust_load_state_dict(model, ckpt_path):
    """Load model weights robustly"""
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {missing}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {unexpected}")


def compose_transforms(T1, T2):
    """Compose two SE3 transforms: result = T1 @ T2"""
    if T1.shape == (3, 4):
        T1 = np.vstack([T1, [0, 0, 0, 1]])
    if T2.shape == (3, 4):
        T2 = np.vstack([T2, [0, 0, 0, 1]])
    result = T1 @ T2
    return result[:3, :]


def rotation_magnitude(T):
    """Compute rotation magnitude in degrees from SE3 transform"""
    if T.shape == (3, 4):
        R = T[:, :3]
    else:
        R = T[:3, :3]
    trace = np.trace(R)
    rot_rad = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(rot_rad * 180.0 / np.pi)


def translation_magnitude(T):
    """Compute translation magnitude from SE3 transform"""
    if T.shape == (3, 4):
        t = T[:, 3]
    else:
        t = T[:3, 3]
    return float(np.linalg.norm(t))


def process_sequence(args):
    """Main sequence processing function"""
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load frames (支持固定点数和可变点数两种格式)
    print(f"\nLoading data from: {args.h5}")
    with h5py.File(args.h5, 'r') as f:
        variable_points = f.attrs.get('variable_points', False)
        
        if variable_points:
            # 可变点数格式
            n_frames = f.attrs['num_frames']
            point_counts = f.attrs.get('point_counts', [])
            data = []
            normals = []
            for i in range(n_frames):
                data.append(np.asarray(f[f'data_{i}']))
                normals.append(np.asarray(f[f'normal_{i}']))
            print(f"Loaded {n_frames} frames with variable points:")
            print(f"  Point counts: {point_counts}")
        else:
            # 固定点数格式
            data = [np.asarray(f['data'][i]) for i in range(len(f['data']))]
            normals = [np.asarray(f['normal'][i]) for i in range(len(f['normal']))]
            n_frames = len(data)
            print(f"Loaded {n_frames} frames with {data[0].shape[0]} points each")
    
    # Load timestamps if available
    h5_dir = Path(args.h5).parent
    timestamps_path = h5_dir / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        print(f"Loaded timestamps from {timestamps_path}")
    else:
        timestamps = np.arange(n_frames, dtype=float)
        print("No timestamps found, using frame indices")
    
    # Build model
    print(f"\nLoading RPM-Net model from: {args.resume}")
    if args.auto_radius:
        # Estimate radius from first pair
        est_r = auto_radius(data[0], data[min(1, n_frames-1)])
        print(f"Auto-estimated feature radius: {est_r:.3f}")
        feat_radius = est_r
    else:
        feat_radius = args.radius
        print(f"Using specified feature radius: {feat_radius:.3f}")
    
    rpm_args = SimpleNamespace(
        method='rpmnet',
        features=['ppf', 'dxyz', 'xyz'],
        feat_dim=96,
        radius=feat_radius,
        num_neighbors=args.neighbors,
        num_reg_iter=args.num_iter,
        add_slack=True,
        resume=args.resume,
        no_slack=False,
        num_sk_iter=5
    )
    
    model = rpmnet_mod.get_model(rpm_args)
    model.to(device)
    model.eval()
    robust_load_state_dict(model, args.resume)
    print("Model loaded successfully")
    
    # Generate frame pairs
    if args.start_frame is not None and args.end_frame is not None:
        frame_range = range(args.start_frame, args.end_frame)
    else:
        frame_range = range(n_frames - 1)
    
    pairs = [(i, i + args.stride) for i in frame_range 
             if i + args.stride < n_frames]
    
    print(f"\n{'='*60}")
    print(f"Processing {len(pairs)} frame pairs with stride {args.stride}")
    print(f"{'='*60}\n")
    
    # Process pairs
    results = []
    accumulated_transform = np.eye(4)[:3, :]  # Identity in (3,4) format
    trajectory = [np.zeros(3)]  # Start at origin
    
    for src_idx, ref_idx in tqdm(pairs, desc="Processing pairs"):
        # Prepare data (已经是list格式，直接使用)
        xyz_src = np.asarray(data[src_idx]).astype(np.float32)
        xyz_ref = np.asarray(data[ref_idx]).astype(np.float32)
        n_src = np.asarray(normals[src_idx]).astype(np.float32)
        n_ref = np.asarray(normals[ref_idx]).astype(np.float32)
        
        pts_src6 = np.concatenate([xyz_src, n_src], axis=-1)[None, ...]
        pts_ref6 = np.concatenate([xyz_ref, n_ref], axis=-1)[None, ...]
        
        batch = {
            'points_src': to_tensor(pts_src6, device),
            'points_ref': to_tensor(pts_ref6, device)
        }
        
        # Run inference
        t_start = time.time()
        with torch.no_grad():
            transforms, endpoints = model(batch, rpm_args.num_reg_iter)
        inference_time = time.time() - t_start
        
        # Extract result
        T_pred = transforms[-1][0].detach().cpu().numpy()  # (3, 4)
        
        # Compute metrics
        rot_deg = rotation_magnitude(T_pred)
        trans_mag = translation_magnitude(T_pred)
        
        # Accumulate transform for trajectory
        accumulated_transform = compose_transforms(accumulated_transform, T_pred)
        current_position = accumulated_transform[:, 3]
        trajectory.append(current_position.copy())
        
        # Store result
        result = {
            'src_frame': int(src_idx),
            'ref_frame': int(ref_idx),
            'src_timestamp': float(timestamps[src_idx]),
            'ref_timestamp': float(timestamps[ref_idx]),
            'transform': T_pred.tolist(),
            'rotation_deg': float(rot_deg),
            'translation_m': float(trans_mag),
            'inference_time_ms': float(inference_time * 1000),
            'accumulated_position': current_position.tolist()
        }
        results.append(result)
        
        if args.verbose:
            print(f"Pair ({src_idx:3d} → {ref_idx:3d}): "
                  f"rot={rot_deg:6.2f}°, trans={trans_mag:6.3f}m, "
                  f"time={inference_time*1000:5.1f}ms")
    
    # Save results
    results_json = os.path.join(args.output_dir, 'sequence_results.json')
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_json}")
    
    # Save trajectory
    trajectory_npy = os.path.join(args.output_dir, 'trajectory.npy')
    trajectory = np.array(trajectory)
    np.save(trajectory_npy, trajectory)
    print(f"Trajectory saved to: {trajectory_npy}")
    
    # Save all transforms
    transforms_npy = os.path.join(args.output_dir, 'pairwise_transforms.npy')
    transforms_array = np.array([r['transform'] for r in results])
    np.save(transforms_npy, transforms_array)
    print(f"Transforms saved to: {transforms_npy}")
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    rotations = [r['rotation_deg'] for r in results]
    translations = [r['translation_m'] for r in results]
    times = [r['inference_time_ms'] for r in results]
    
    print(f"Rotations (deg):")
    print(f"  Mean: {np.mean(rotations):6.2f}, Std: {np.std(rotations):6.2f}")
    print(f"  Min:  {np.min(rotations):6.2f}, Max: {np.max(rotations):6.2f}")
    
    print(f"\nTranslations (m):")
    print(f"  Mean: {np.mean(translations):6.3f}, Std: {np.std(translations):6.3f}")
    print(f"  Min:  {np.min(translations):6.3f}, Max: {np.max(translations):6.3f}")
    
    print(f"\nInference time (ms):")
    print(f"  Mean: {np.mean(times):6.1f}, Std: {np.std(times):6.1f}")
    print(f"  Min:  {np.min(times):6.1f}, Max: {np.max(times):6.1f}")
    
    print(f"\nTrajectory length: {np.linalg.norm(trajectory[-1] - trajectory[0]):.2f}m")
    
    # Visualize trajectory if requested
    if args.visualize_trajectory:
        visualize_trajectory_3d(trajectory, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Sequence processing complete!")
    print(f"{'='*60}")


def visualize_trajectory_3d(trajectory, output_dir):
    """Visualize and save 3D trajectory"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-o', markersize=3, linewidth=1.5, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                c='g', marker='o', s=100, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                c='r', marker='*', s=200, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # XY plot (top view)
    ax2 = fig.add_subplot(222)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', markersize=3, linewidth=1.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='g', marker='o', s=100, label='Start')
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', marker='*', s=200, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # XZ plot (side view)
    ax3 = fig.add_subplot(223)
    ax3.plot(trajectory[:, 0], trajectory[:, 2], 'b-o', markersize=3, linewidth=1.5)
    ax3.scatter(trajectory[0, 0], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 2], c='r', marker='*', s=200, label='End')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # YZ plot (front view)
    ax4 = fig.add_subplot(224)
    ax4.plot(trajectory[:, 1], trajectory[:, 2], 'b-o', markersize=3, linewidth=1.5)
    ax4.scatter(trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    ax4.scatter(trajectory[-1, 1], trajectory[-1, 2], c='r', marker='*', s=200, label='End')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (YZ)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'trajectory_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory visualization saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Process radar frame sequence for odometry")
    parser.add_argument('--h5', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--resume', type=str, required=True, help='Path to RPMNet checkpoint')
    parser.add_argument('--output_dir', type=str, default='sequence_results/',
                       help='Directory to save results')
    
    # Frame selection
    parser.add_argument('--stride', type=int, default=1,
                       help='Process every Nth frame (default: 1 = consecutive pairs)')
    parser.add_argument('--start_frame', type=int, default=None,
                       help='Start processing from this frame')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='End processing at this frame')
    
    # Model parameters
    parser.add_argument('--num_iter', type=int, default=5,
                       help='Number of RPMNet iterations')
    parser.add_argument('--auto_radius', action='store_true',
                       help='Auto-estimate feature radius')
    parser.add_argument('--radius', type=float, default=2.0,
                       help='Feature radius if not auto-estimated')
    parser.add_argument('--neighbors', type=int, default=30,
                       help='Number of neighbors for features')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    # Output options
    parser.add_argument('--visualize_trajectory', action='store_true',
                       help='Generate trajectory visualization plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each pair')
    
    args = parser.parse_args()
    
    process_sequence(args)


if __name__ == '__main__':
    main()

