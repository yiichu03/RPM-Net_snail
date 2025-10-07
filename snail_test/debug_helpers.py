"""Debug helper utilities for understanding RPM-Net internals

These functions help you understand what's happening inside RPM-Net during
inference, useful for stepping through with a debugger or analyzing intermediate results.

Usage (in debugger or notebook):
    from debug_helpers import inspect_data_batch, inspect_features, inspect_sinkhorn
"""
import numpy as np
import torch


def inspect_data_batch(data_dict):
    """Inspect the data dictionary passed to RPM-Net
    
    Args:
        data_dict: Dictionary with 'points_src' and 'points_ref'
    """
    print("="*60)
    print("Data Batch Inspection")
    print("="*60)
    
    for key in data_dict:
        val = data_dict[key]
        if isinstance(val, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            print(f"  Device: {val.device}")
            print(f"  Range: [{val.min().item():.3f}, {val.max().item():.3f}]")
            print(f"  Mean: {val.mean().item():.3f}")
            print(f"  Std: {val.std().item():.3f}")
        else:
            print(f"{key}: {type(val)}")
    
    # Check if points and normals are separate
    if 'points_src' in data_dict:
        pts_src = data_dict['points_src']
        if pts_src.shape[-1] == 6:
            xyz = pts_src[..., :3]
            normals = pts_src[..., 3:6]
            print("\nSource points (xyz):")
            print(f"  Range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
            print(f"  Centroid: {xyz.mean(dim=1)}")
            print("\nSource normals:")
            print(f"  Range: [{normals.min().item():.3f}, {normals.max().item():.3f}]")
            norm_magnitude = torch.norm(normals, dim=-1).mean()
            print(f"  Mean magnitude: {norm_magnitude.item():.6f}")
            if abs(norm_magnitude.item() - 1.0) > 0.01:
                print(f"  ⚠️  Warning: Normals don't appear to be unit vectors!")


def inspect_features(feat_src, feat_ref, feature_names=['ppf', 'dxyz', 'xyz']):
    """Inspect extracted features
    
    Args:
        feat_src: Source features (B, J, C)
        feat_ref: Reference features (B, K, C)
        feature_names: List of feature types
    """
    print("\n" + "="*60)
    print("Feature Inspection")
    print("="*60)
    
    print(f"\nSource features: {feat_src.shape}")
    print(f"Reference features: {feat_ref.shape}")
    
    print(f"\nSource feature stats:")
    print(f"  Mean: {feat_src.mean().item():.3f}")
    print(f"  Std: {feat_src.std().item():.3f}")
    print(f"  Range: [{feat_src.min().item():.3f}, {feat_src.max().item():.3f}]")
    
    print(f"\nReference feature stats:")
    print(f"  Mean: {feat_ref.mean().item():.3f}")
    print(f"  Std: {feat_ref.std().item():.3f}")
    print(f"  Range: [{feat_ref.min().item():.3f}, {feat_ref.max().item():.3f}]")
    
    # Check if features have reasonable scale
    if feat_src.std().item() < 0.01:
        print(f"  ⚠️  Warning: Features have very low variance!")
    if feat_src.abs().max().item() > 100:
        print(f"  ⚠️  Warning: Features have very large values!")


def inspect_affinity_matrix(affinity, beta, alpha):
    """Inspect affinity matrix before Sinkhorn
    
    Args:
        affinity: Log affinity matrix (B, J, K)
        beta: Temperature parameter (B,)
        alpha: Threshold parameter (B,)
    """
    print("\n" + "="*60)
    print("Affinity Matrix Inspection")
    print("="*60)
    
    print(f"\nAffinity shape: {affinity.shape}")
    print(f"Beta: {beta}")
    print(f"Alpha: {alpha}")
    
    print(f"\nAffinity stats:")
    print(f"  Mean: {affinity.mean().item():.3f}")
    print(f"  Std: {affinity.std().item():.3f}")
    print(f"  Range: [{affinity.min().item():.3f}, {affinity.max().item():.3f}]")
    
    # Check sparsity (how many values are near zero in log space)
    exp_affinity = torch.exp(affinity)
    threshold = 0.01
    sparse_count = (exp_affinity < threshold).sum().item()
    total = exp_affinity.numel()
    print(f"\nSparsity (exp < {threshold}): {sparse_count}/{total} ({100*sparse_count/total:.1f}%)")


def inspect_sinkhorn_output(perm_matrix):
    """Inspect Sinkhorn output (correspondence matrix)
    
    Args:
        perm_matrix: Permutation/correspondence matrix (B, J, K)
    """
    print("\n" + "="*60)
    print("Sinkhorn Output Inspection")
    print("="*60)
    
    print(f"\nPermutation matrix shape: {perm_matrix.shape}")
    
    # Check doubly stochastic properties
    row_sums = perm_matrix.sum(dim=2)
    col_sums = perm_matrix.sum(dim=1)
    
    print(f"\nRow sums (should be ≤1):")
    print(f"  Mean: {row_sums.mean().item():.6f}")
    print(f"  Min: {row_sums.min().item():.6f}")
    print(f"  Max: {row_sums.max().item():.6f}")
    
    print(f"\nColumn sums (should be ≤1):")
    print(f"  Mean: {col_sums.mean().item():.6f}")
    print(f"  Min: {col_sums.min().item():.6f}")
    print(f"  Max: {col_sums.max().item():.6f}")
    
    # Check sparsity
    threshold = 0.01
    sparse_count = (perm_matrix < threshold).sum().item()
    total = perm_matrix.numel()
    print(f"\nSparsity (values < {threshold}): {sparse_count}/{total} ({100*sparse_count/total:.1f}%)")
    
    # Find strongest matches
    max_per_row, max_indices = perm_matrix.max(dim=2)
    print(f"\nStrongest matches per source point:")
    print(f"  Mean strength: {max_per_row.mean().item():.6f}")
    print(f"  Min strength: {max_per_row.min().item():.6f}")
    print(f"  Max strength: {max_per_row.max().item():.6f}")
    
    # Count confident matches
    confident_threshold = 0.5
    confident_matches = (max_per_row > confident_threshold).sum().item()
    print(f"\nConfident matches (>{confident_threshold}): {confident_matches}/{perm_matrix.shape[1]} "
          f"({100*confident_matches/perm_matrix.shape[1]:.1f}%)")


def inspect_transform(transform):
    """Inspect computed SE3 transform
    
    Args:
        transform: SE3 transform matrix (B, 3, 4) or (3, 4)
    """
    print("\n" + "="*60)
    print("Transform Inspection")
    print("="*60)
    
    if transform.dim() == 3:
        T = transform[0]  # Take first batch element
    else:
        T = transform
    
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    
    R = T[:, :3]
    t = T[:, 3]
    
    print("\nRotation matrix R:")
    print(R)
    
    print("\nTranslation vector t:")
    print(t)
    
    # Check rotation matrix properties
    det = np.linalg.det(R)
    print(f"\nRotation matrix determinant: {det:.6f}")
    if abs(det - 1.0) > 0.01:
        print(f"  ⚠️  Warning: Determinant should be 1.0 for proper rotation!")
    
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    print(f"Orthogonality error: {orthogonality_error:.6e}")
    if orthogonality_error > 1e-4:
        print(f"  ⚠️  Warning: Matrix is not orthogonal!")
    
    # Compute rotation angle
    trace = np.trace(R)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    angle_deg = angle_rad * 180 / np.pi
    print(f"\nRotation angle: {angle_deg:.2f}°")
    
    # Compute translation magnitude
    trans_mag = np.linalg.norm(t)
    print(f"Translation magnitude: {trans_mag:.3f}")


def inspect_iteration_progress(transforms_list):
    """Inspect how transform evolves across iterations
    
    Args:
        transforms_list: List of transforms from each iteration
    """
    print("\n" + "="*60)
    print("Iteration Progress Inspection")
    print("="*60)
    
    print(f"\nNumber of iterations: {len(transforms_list)}")
    
    for i, T in enumerate(transforms_list):
        if isinstance(T, torch.Tensor):
            T = T[0].detach().cpu().numpy()  # Take first batch element
        
        R = T[:, :3]
        t = T[:, 3]
        
        # Compute rotation angle
        trace = np.trace(R)
        angle_deg = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)) * 180 / np.pi
        
        # Compute translation magnitude
        trans_mag = np.linalg.norm(t)
        
        print(f"\nIteration {i}:")
        print(f"  Rotation: {angle_deg:.2f}°")
        print(f"  Translation: {trans_mag:.3f}m")
    
    # Check convergence
    if len(transforms_list) > 1:
        print("\nConvergence analysis:")
        for i in range(1, len(transforms_list)):
            T_prev = transforms_list[i-1]
            T_curr = transforms_list[i]
            
            if isinstance(T_prev, torch.Tensor):
                T_prev = T_prev[0].detach().cpu().numpy()
            if isinstance(T_curr, torch.Tensor):
                T_curr = T_curr[0].detach().cpu().numpy()
            
            # Compute change
            R_change = np.linalg.norm(T_curr[:, :3] - T_prev[:, :3], 'fro')
            t_change = np.linalg.norm(T_curr[:, 3] - T_prev[:, 3])
            
            print(f"  Iter {i-1}→{i}: ΔR={R_change:.6f}, Δt={t_change:.6f}")


def print_debug_breakpoint_guide():
    """Print a guide for setting breakpoints in RPM-Net"""
    print("="*60)
    print("RPM-Net Debugging Guide")
    print("="*60)
    
    print("\nRecommended breakpoints in src/models/rpmnet.py:")
    print("  Line 184: Start of registration loop")
    print("  Line 186: After weight prediction (check beta, alpha)")
    print("  Line 187-188: After feature extraction (check feat_src, feat_ref)")
    print("  Line 191: After affinity computation (check affinity matrix)")
    print("  Line 194: After Sinkhorn (check perm_matrix)")
    print("  Line 199: After SVD (check computed transform)")
    print("  Line 200: After applying transform (check xyz_src_t)")
    
    print("\nUseful inspection calls:")
    print("  inspect_data_batch(data)")
    print("  inspect_features(feat_src, feat_ref)")
    print("  inspect_affinity_matrix(affinity, beta, alpha)")
    print("  inspect_sinkhorn_output(perm_matrix)")
    print("  inspect_transform(transform)")
    
    print("\nQuick debug session example:")
    print("  1. Set breakpoint at line 184 (start of loop)")
    print("  2. Run inference script")
    print("  3. When breakpoint hits, in debug console:")
    print("     from debug_helpers import *")
    print("     inspect_data_batch(data)")
    print("  4. Step through iterations and inspect intermediate results")


if __name__ == '__main__':
    print_debug_breakpoint_guide()

