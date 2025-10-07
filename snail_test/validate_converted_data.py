"""Validation script for converted radar data

This script helps verify that the HDF5 conversion is correct and provides
statistics about the data quality.

Usage:
    python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5
"""
import argparse
import h5py
import numpy as np
import open3d as o3d
from pathlib import Path


def validate_h5_structure(h5_path: str):
    """Validate HDF5 file structure and contents"""
    print("=" * 60)
    print("HDF5 Structure Validation")
    print("=" * 60)
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nFile: {h5_path}")
        print(f"Keys: {list(f.keys())}")
        
        # Check required datasets
        required = ['data', 'normal', 'label']
        for key in required:
            if key not in f:
                print(f"❌ Missing required dataset: {key}")
                return False
            else:
                print(f"✓ Found dataset: {key}")
        
        # Check shapes
        data = f['data'][:]
        normals = f['normal'][:]
        labels = f['label'][:]
        
        print(f"\nData shape: {data.shape}")
        print(f"Normal shape: {normals.shape}")
        print(f"Label shape: {labels.shape}")
        
        n_frames, n_points, dim = data.shape
        
        if dim != 3:
            print(f"❌ Data dimension should be 3, got {dim}")
            return False
        
        if normals.shape != (n_frames, n_points, 3):
            print(f"❌ Normal shape mismatch")
            return False
        
        print(f"\n✓ Structure validation passed")
        print(f"  Frames: {n_frames}")
        print(f"  Points per frame: {n_points}")
        
        return True, data, normals, labels


def analyze_point_cloud_stats(data: np.ndarray):
    """Analyze point cloud statistics"""
    print("\n" + "=" * 60)
    print("Point Cloud Statistics")
    print("=" * 60)
    
    n_frames = data.shape[0]
    
    for i in range(min(n_frames, 5)):  # Show stats for first 5 frames
        points = data[i]
        
        print(f"\nFrame {i}:")
        print(f"  Centroid: {np.mean(points, axis=0)}")
        print(f"  Std dev: {np.std(points, axis=0)}")
        print(f"  Min: {np.min(points, axis=0)}")
        print(f"  Max: {np.max(points, axis=0)}")
        
        # Compute bounding box size
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        print(f"  Bounding box size: {bbox_size}")
        
        # Compute average nearest neighbor distance
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_nn_dist = np.mean(distances[:, 1])  # Skip self (distance=0)
        print(f"  Avg nearest neighbor distance: {avg_nn_dist:.4f}")
        
        # Compute point cloud extent (for radius estimation)
        extent = np.linalg.norm(bbox_size)
        print(f"  Point cloud extent: {extent:.2f}")
        print(f"  Suggested feature radius: {avg_nn_dist * 5:.3f} - {avg_nn_dist * 10:.3f}")


def analyze_normal_quality(normals: np.ndarray, h5_path: str):
    """Analyze normal vector quality"""
    print("\n" + "=" * 60)
    print("Normal Quality Analysis")
    print("=" * 60)
    
    # Check if normal_is_random.npy exists
    h5_dir = Path(h5_path).parent
    random_mask_path = h5_dir / "normal_is_random.npy"
    
    if random_mask_path.exists():
        random_mask = np.load(random_mask_path)
        print(f"\nRandom normal mask found: {random_mask_path}")
        print(f"Shape: {random_mask.shape}")
        
        for i in range(min(normals.shape[0], 5)):
            n_random = random_mask[i].sum()
            pct_random = 100.0 * n_random / random_mask.shape[1]
            print(f"Frame {i}: {n_random}/{random_mask.shape[1]} random ({pct_random:.1f}%)")
            
            if pct_random > 50:
                print(f"  ⚠️  WARNING: >50% random normals, may affect registration quality")
            elif pct_random > 30:
                print(f"  ⚠️  Caution: >30% random normals")
            else:
                print(f"  ✓ Good: <30% random normals")
    else:
        print(f"\nNo random mask found at {random_mask_path}")
    
    # Check normal magnitudes
    print(f"\nNormal magnitude analysis:")
    for i in range(min(normals.shape[0], 5)):
        norms = np.linalg.norm(normals[i], axis=1)
        print(f"Frame {i}:")
        print(f"  Mean magnitude: {np.mean(norms):.6f}")
        print(f"  Std magnitude: {np.std(norms):.6f}")
        print(f"  Min/Max magnitude: {np.min(norms):.6f} / {np.max(norms):.6f}")
        
        # Check if normalized
        if np.abs(np.mean(norms) - 1.0) < 0.01:
            print(f"  ✓ Normals appear to be normalized")
        else:
            print(f"  ⚠️  Normals may not be properly normalized")


def visualize_frame(data: np.ndarray, normals: np.ndarray, frame_idx: int, save_path: str = None):
    """Visualize a single frame with normals"""
    print(f"\n" + "=" * 60)
    print(f"Visualizing Frame {frame_idx}")
    print("=" * 60)
    
    points = data[frame_idx]
    norms = normals[frame_idx]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(norms)
    
    # Color by height (Z coordinate)
    colors = np.zeros_like(points)
    z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-8)
    colors[:, 0] = z_normalized  # Red channel
    colors[:, 2] = 1.0 - z_normalized  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved visualization to: {save_path}")
    else:
        print("Displaying point cloud (close window to continue)...")
        o3d.visualization.draw_geometries([pcd], 
                                          window_name=f"Frame {frame_idx}",
                                          point_show_normal=True)


def check_frame_pair_overlap(data: np.ndarray, idx1: int, idx2: int, threshold: float = 5.0):
    """Estimate overlap between two frames"""
    print(f"\n" + "=" * 60)
    print(f"Frame Pair Overlap Analysis: {idx1} ↔ {idx2}")
    print("=" * 60)
    
    from sklearn.neighbors import NearestNeighbors
    
    pts1 = data[idx1]
    pts2 = data[idx2]
    
    # Find nearest neighbors from pts1 to pts2
    nbrs = NearestNeighbors(n_neighbors=1).fit(pts2)
    distances, _ = nbrs.kneighbors(pts1)
    
    overlap_count = np.sum(distances.flatten() < threshold)
    overlap_pct = 100.0 * overlap_count / len(pts1)
    
    print(f"Distance threshold: {threshold}")
    print(f"Overlapping points from frame {idx1}: {overlap_count}/{len(pts1)} ({overlap_pct:.1f}%)")
    print(f"Mean nearest distance: {np.mean(distances):.3f}")
    print(f"Median nearest distance: {np.median(distances):.3f}")
    
    if overlap_pct < 30:
        print(f"⚠️  WARNING: Low overlap (<30%), registration may be difficult")
    elif overlap_pct < 50:
        print(f"⚠️  Caution: Moderate overlap (30-50%)")
    else:
        print(f"✓ Good overlap (>50%)")
    
    return overlap_pct


def main():
    parser = argparse.ArgumentParser(description="Validate converted radar data")
    parser.add_argument('--h5', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--visualize', type=int, default=None, 
                       help='Frame index to visualize (default: None)')
    parser.add_argument('--save_vis', type=str, default=None,
                       help='Save visualization to this path instead of displaying')
    parser.add_argument('--check_overlap', nargs=2, type=int, default=None,
                       help='Check overlap between two frames (e.g., --check_overlap 0 1)')
    parser.add_argument('--overlap_threshold', type=float, default=5.0,
                       help='Distance threshold for overlap check (default: 5.0)')
    
    args = parser.parse_args()
    
    # Validate structure
    result = validate_h5_structure(args.h5)
    if not result:
        return
    
    valid, data, normals, labels = result
    
    # Analyze statistics
    analyze_point_cloud_stats(data)
    
    # Analyze normals
    analyze_normal_quality(normals, args.h5)
    
    # Visualize if requested
    if args.visualize is not None:
        if args.visualize >= len(data):
            print(f"\n❌ Frame {args.visualize} out of range (0-{len(data)-1})")
        else:
            visualize_frame(data, normals, args.visualize, args.save_vis)
    
    # Check overlap if requested
    if args.check_overlap is not None:
        idx1, idx2 = args.check_overlap
        if idx1 >= len(data) or idx2 >= len(data):
            print(f"\n❌ Frame indices out of range (0-{len(data)-1})")
        else:
            check_frame_pair_overlap(data, idx1, idx2, args.overlap_threshold)
    
    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print("\nRecommendations:")
    print("1. If random normals >30%, consider increasing k_normal in conversion")
    print("2. Use suggested feature radius range for RPM-Net inference")
    print("3. Test with frame pairs that have >50% overlap first")
    print("4. Visualize frames to ensure coordinate system is correct")


if __name__ == '__main__':
    main()

