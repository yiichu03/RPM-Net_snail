from pathlib import Path
import sys

# infer_single_pair.py 位于 .../RPMNet/snail_test/
# 我们要把 .../RPMNet/src 加到 PYTHONPATH
SRC_DIR = Path(__file__).resolve().parents[1] / "src"   # parents[1] == RPMNet
sys.path.insert(0, str(SRC_DIR))

import argparse
import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import open3d as o3d
'''
# 选第0帧做 source，第1帧做 ref；自动估半径；保存可视化
python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5   --resume D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth   --src 0 --ref 3 --num_iter 10 --auto_radius --neighbors 50 --save_vis
python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5  --resume D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth  --src 0 --ref 3 --num_iter 10 --auto_radius --neighbors 50 --save_vis
'''
# --- utils -------------------------------------------------------------------

def to_tensor(x, device):
    return torch.from_numpy(x).float().to(device)

def apply_se3(xyz, T):
    """xyz: (N,3) numpy, T: (3,4) or (4,4) torch/numpy -> (N,3) numpy"""
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
    """Estimate a reasonable feature radius from kNN distances (percentile over both clouds)."""
    import sklearn.neighbors as skn
    pts = np.vstack([points_src, points_ref])
    nbrs = skn.NearestNeighbors(n_neighbors=min(k+1, len(pts))).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    # skip self (0), take median of k-th neighbor
    base = np.median(dists[:, 1:])
    # a slightly conservative enlargement
    return float(np.percentile(dists[:, 1:], pct)) if np.isfinite(base) else 1.0


def flip_normals_outward(xyz, normals, origin=(0.0, 0.0, 0.0)):
    o = np.asarray(origin, dtype=np.float32)[None, :]
    v = xyz - o
    flip = (np.sum(v * normals, axis=1) < 0)
    normals[flip] = -normals[flip]
    return normals


def robust_load_state_dict(model, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # strip possible 'module.' prefixes
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[ckpt] loaded from {ckpt_path}")
    if missing:
        print("[ckpt] missing keys:", missing)
    if unexpected:
        print("[ckpt] unexpected keys:", unexpected)


# --- main inference ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("RPMNet single-pair inference on SNAIL radar frames")
    parser.add_argument('--h5', type=str, required=True, help='radar_single_frames_test0.h5 path')
    parser.add_argument('--resume', type=str, required=True, help='RPMNet checkpoint .pth')
    parser.add_argument('--src', type=int, default=0, help='source frame index in H5')
    parser.add_argument('--ref', type=int, default=1, help='reference frame index in H5')
    parser.add_argument('--num_iter', type=int, default=5, help='RPMNet registration iterations')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--auto_radius', action='store_true', help='estimate feature radius from data')
    parser.add_argument('--radius', type=float, default=2.0, help='feature radius (ignored if --auto_radius)')
    parser.add_argument('--neighbors', type=int, default=30, help='neighbors per point for features')
    parser.add_argument('--align_normals', action='store_true', help='flip normals to face outward from origin')
    parser.add_argument('--save_vis', action='store_true', help='save before/after PLY into same folder as H5')

    args = parser.parse_args()

    # device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')

    # load frames (支持固定点数和可变点数两种格式)
    with h5py.File(args.h5, 'r') as f:
        # 检查是否是可变点数格式
        variable_points = f.attrs.get('variable_points', False)
        
        if variable_points:
            # 可变点数格式：每帧单独存储
            num_frames = f.attrs['num_frames']
            assert args.src < num_frames and args.ref < num_frames, "src/ref index out of range"
            
            xyz_src = np.asarray(f[f'data_{args.src}']).astype(np.float32)
            xyz_ref = np.asarray(f[f'data_{args.ref}']).astype(np.float32)
            n_src = np.asarray(f[f'normal_{args.src}']).astype(np.float32)
            n_ref = np.asarray(f[f'normal_{args.ref}']).astype(np.float32)
            
            print(f"Loaded variable-size frames:")
            print(f"  Source frame {args.src}: {len(xyz_src)} points")
            print(f"  Reference frame {args.ref}: {len(xyz_ref)} points")
        else:
            # 固定点数格式：标准数组
            data = np.asarray(f['data'])           # (F, N, 3)
            normals = np.asarray(f['normal'])      # (F, N, 3)
            assert args.src < len(data) and args.ref < len(data), "src/ref index out of range"

            xyz_src = data[args.src].astype(np.float32)
            xyz_ref = data[args.ref].astype(np.float32)
            n_src = normals[args.src].astype(np.float32)
            n_ref = normals[args.ref].astype(np.float32)
            
            print(f"Loaded fixed-size frames: {len(xyz_src)} points each")

    if args.align_normals:
        n_src = flip_normals_outward(xyz_src, n_src)
        n_ref = flip_normals_outward(xyz_ref, n_ref)

    # build model args namespace (keep in sync with your RPMNet implementation)
    # Using EarlyFusion defaults: features include xyz, dxyz and ppf
    if args.auto_radius:
        est_r = auto_radius(xyz_src, xyz_ref)
        print(f"[auto] feature radius ~= {est_r:.3f}")
        feat_radius = est_r
    else:
        feat_radius = args.radius

    rpm_args = SimpleNamespace(
        method='rpmnet',
        features=['ppf', 'dxyz', 'xyz'],
        feat_dim=96,
        radius=feat_radius,
        num_neighbors=args.neighbors,
        num_reg_iter=args.num_iter,
        add_slack=True,             # typical setting in RPMNet
        resume=args.resume,
        no_slack=False,
        num_sk_iter=5
    )

    # import your project modules (assumes this script is launched from project root)
    import models.rpmnet as rpmnet_mod

    model = rpmnet_mod.get_model(rpm_args)
    model.to(device)
    model.eval()

    # robust weight loading (bypass CheckPointManager to keep script standalone)
    robust_load_state_dict(model, args.resume)

    # assemble data dict expected by RPMNet forward
    pts_src6 = np.concatenate([xyz_src, n_src], axis=-1)[None, ...]   # (1, 1024, 6)
    pts_ref6 = np.concatenate([xyz_ref, n_ref], axis=-1)[None, ...]   # (1, 1024, 6)

    batch = {
        'points_src': to_tensor(pts_src6, device),
        'points_ref': to_tensor(pts_ref6, device)
    }

    # Prepare output directory
    out_dir = os.path.dirname(os.path.abspath(args.h5))

    with torch.no_grad():
        transforms, endpoints = model(batch, rpm_args.num_reg_iter)
    
    # Save detailed results for visualization
    if args.save_vis:
        import pickle
        from scipy import sparse
        
        # Save transforms in eval format: (1, n_iter, 3, 4)
        transforms_np = torch.stack(transforms, dim=1).detach().cpu().numpy()  # (1, n_iter, 3, 4)
        pred_transforms_path = os.path.join(out_dir, 'pred_transforms.npy')
        np.save(pred_transforms_path, transforms_np)
        print(f"Saved pred_transforms.npy: shape {transforms_np.shape}")
        
        # Save permutation matrices in eval format
        if 'perm_matrices' in endpoints:
            perm_matrices = torch.stack(endpoints['perm_matrices'], dim=1).detach().cpu().numpy()  # (1, n_iter, J, K)
            
            # Convert to sparse format (same as eval.py)
            thresh = np.percentile(perm_matrices, 99.9, axis=[2, 3])
            below_thresh_mask = perm_matrices < thresh[:, :, None, None]
            perm_matrices[below_thresh_mask] = 0.0
            
            # Create sparse matrices list
            sparse_perm_matrices_list = []
            for i_iter in range(perm_matrices.shape[1]):
                sparse_perm_matrices_list.append(sparse.coo_matrix(perm_matrices[0, i_iter, :, :]))
            
            # Save as pickle (list of sparse matrices for one sample)
            perm_matrices_path = os.path.join(out_dir, 'perm_matrices.pickle')
            with open(perm_matrices_path, 'wb') as f:
                pickle.dump([sparse_perm_matrices_list], f)  # Wrap in list for consistency
            print(f"Saved perm_matrices.pickle: {len(sparse_perm_matrices_list)} iterations")

    T_last = transforms[-1][0]   # (3,4) torch
    T_np = T_last.detach().cpu().numpy()

    # report
    R = T_np[:, :3]
    t = T_np[:, 3]
    trace = np.trace(R)
    rot_rad = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(rot_rad * 180.0 / np.pi)
    t_norm = float(np.linalg.norm(t))
    print("=== RPMNet result (src -> ref) ===")
    print("R (3x3):\n", R)
    print("t (m): ", t)
    print(f"rotation = {rot_deg:.3f} deg, translation = {t_norm:.3f} m")

    # optional visualization save
    if args.save_vis:
        # Save data dict for visualization (original points without transform)
        data_dict = {
            'points_src': xyz_src,
            'points_ref': xyz_ref,
            'normals_src': n_src,
            'normals_ref': n_ref
        }
        data_dict_path = os.path.join(out_dir, 'data_dict.npy')
        np.save(data_dict_path, data_dict, allow_pickle=True)
        print(f"Saved data_dict.npy for visualization")
        
        # before
        p_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_src))
        p_src.paint_uniform_color([1, 0.2, 0.2])
        p_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ref))
        p_ref.paint_uniform_color([0.2, 0.8, 1])
        o3d.io.write_point_cloud(os.path.join(out_dir, 'pair_before.ply'), p_src + p_ref)
        # after
        xyz_src_aligned = apply_se3(xyz_src, T_np)
        p_src2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_src_aligned))
        p_src2.paint_uniform_color([1, 0.2, 0.2])
        o3d.io.write_point_cloud(os.path.join(out_dir, 'pair_after.ply'), p_src2 + p_ref)
        transforms_np = torch.stack(transforms, dim=1).detach().cpu().numpy()
        np.save(os.path.join(out_dir, 'pred_transforms.npy'), transforms_np)
        T_last = transforms[-1][0].detach().cpu().numpy()   # (3,4)
        np.savetxt(os.path.join(out_dir, f"T_src{args.src}_ref{args.ref}.txt"), T_last)
        print(f"Saved PLY to {out_dir} (pair_before.ply, pair_after.ply)")



if __name__ == '__main__':
    main()
