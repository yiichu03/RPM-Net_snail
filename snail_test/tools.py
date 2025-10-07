# tools.py
from pathlib import Path
import os
import numpy as np
import torch
import open3d as o3d

# 可选：也可以把这些放到函数内部 import
import pickle
from scipy import sparse


def apply_se3(xyz, T):
    """xyz: (N,3) numpy, T: (3,4)  torch/numpy -> (N,3) numpy"""
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    if T.shape == (3, 4):
        R, t = T[:, :3], T[:, 3]
    else:
        raise ValueError("T must be (3,4)")
    return xyz @ R.T + t


def save_inference_visuals(
    out_dir: str,
    transforms,                 # list[Tensor(B,3,4)] 或 Tensor(B,n_iter,3,4)
    endpoints: dict,
    xyz_src: np.ndarray,
    xyz_ref: np.ndarray,
    n_src: np.ndarray,
    n_ref: np.ndarray,
    src_idx: int,
    ref_idx: int,
    save_perm: bool = True,     # 是否保存稀疏化的软匹配矩阵
    save_ply: bool = True,      # 保存前/后的 PLY 点云
    save_data_dict: bool = True,# 保存原始数据字典 numpy
    print_report: bool = True   # 打印 R/t 概要
):
    """
    打包：保存 pred_transforms.npy、(可选) perm_matrices.pickle、(可选) pair_before/after.ply、
         (可选) data_dict.npy、T_src{src}_ref{ref}.txt，并返回关键信息。
    """
    import os
    import pickle
    from pathlib import Path
    from scipy import sparse

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(transforms, torch.Tensor):
        tr_np = transforms.detach().cpu().numpy()
        if tr_np.ndim == 3:              # (B,3,4) → (B,1,3,4)
            tr_np = tr_np[:, None, :, :]
    else:
        # list[Tensor(B,3,4)] → (B,n_iter,3,4)
        tr_np = torch.stack(transforms, dim=1).detach().cpu().numpy()

    np.save(os.path.join(out_dir, 'pred_transforms.npy'), tr_np)

    T_last =  tr_np[0, -1]   # (3,4) torch

    # 2) (可选) 保存软匹配矩阵（稀疏化后）
    perm_pickle_path = None
    if save_perm and ('perm_matrices' in endpoints):
        P_seq = endpoints['perm_matrices']  # list[n_iter] of Tensor(B,J,K) 或 ndarray
        if isinstance(P_seq[0], torch.Tensor):
            P_np = torch.stack(P_seq, dim=1).detach().cpu().numpy()  # (B,n_iter,J,K)
        else:
            P_np = np.stack(P_seq, axis=1)  # (B,n_iter,J,K)

        thresh = np.percentile(P_np, 99.9, axis=[2, 3])
        below_mask = P_np < thresh[:, :, None, None]
        P_np[below_mask] = 0.0

        # 与 eval.py 保持一致的 pickle 结构（外层按 batch；这里 B=1）
        perm_list_batch = []
        B, n_iter = P_np.shape[:2]
        for b in range(B):
            sparse_list_per_b = []
            for i_iter in range(n_iter):
                sparse_list_per_b.append(sparse.coo_matrix(P_np[b, i_iter, :, :]))
            perm_list_batch.append(sparse_list_per_b)

        perm_pickle_path = os.path.join(out_dir, 'perm_matrices.pickle')
        with open(perm_pickle_path, 'wb') as f:
            # 和 eval.py 一样：pickle.dump(list_of_list_of_sparse)
            pickle.dump(perm_list_batch, f)  # 外层 list 是 batch 维度
        print(f"Saved perm_matrices.pickle: {len(perm_list_batch[0])} iterations, batch={len(perm_list_batch)}")

    # 3) (可选) 保存数据字典（便于可视化或复现）
    data_dict_path = None
    if save_data_dict:
        data_dict = {
            'points_src': xyz_src, 'normals_src': n_src,
            'points_ref': xyz_ref, 'normals_ref': n_ref
        }
        data_dict_path = os.path.join(out_dir, 'data_dict.npy')
        np.save(data_dict_path, data_dict, allow_pickle=True)

    # 4) (可选) 保存配准前/后的 PLY
    ply_before_path = ply_after_path = None
    if save_ply:
        p_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_src))
        p_src.paint_uniform_color([1, 0.2, 0.2])
        p_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ref))
        p_ref.paint_uniform_color([0.2, 0.8, 1])
        ply_before_path = os.path.join(out_dir, 'pair_before.ply')
        o3d.io.write_point_cloud(ply_before_path, p_src + p_ref)

        xyz_src_aligned = apply_se3(xyz_src, T_last)
        p_src2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_src_aligned))
        p_src2.paint_uniform_color([1, 0.2, 0.2])
        ply_after_path = os.path.join(out_dir, 'pair_after.ply')
        o3d.io.write_point_cloud(ply_after_path, p_src2 + p_ref)

    # 5) 保存最后一次 T 为 txt（便于快速查看）
    T_txt_path = os.path.join(out_dir, f"T_src{src_idx}_ref{ref_idx}.txt")
    np.savetxt(T_txt_path, T_last)

    # 6) (可选) 打印概要
    if print_report:
        R = T_last[:, :3]; t = T_last[:, 3]
        trace = np.trace(R)
        rot_rad = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
        rot_deg = float(rot_rad * 180.0 / np.pi)
        t_norm = float(np.linalg.norm(t))
        print("=== RPMNet result (src -> ref) ===")
        print("R (3x3):\n", R)
        print("t (m): ", t)
        print(f"rotation = {rot_deg:.3f} deg, translation = {t_norm:.3f} m")

    return {
        'pred_transforms_npy': os.path.join(out_dir, 'pred_transforms.npy'),
        'perm_pickle': perm_pickle_path,
        'data_dict_npy': data_dict_path,
        'ply_before': ply_before_path,
        'ply_after': ply_after_path,
        'T_txt': T_txt_path,
        'T_last': T_last
    }

def estimate_normals_for_radar(points: np.ndarray, k: int = 20):
    """
    用 Open3D估计法向量。
    k: KNN邻居数。
    返回:
      normals: (N,3) float32
      is_bad_mask: (N,) bool（非有限或范数极小的法线）
    """
    N = points.shape[0]
    if N == 0:
        return np.empty((0, 3), np.float32), np.empty((0,), bool)
    if N < 3:
        normals = np.zeros((N, 3), np.float32)
        is_bad_mask = np.ones(N, dtype=bool)
        print(f"[estimate_normals] N={N} < 3 → all bad")
        return normals, is_bad_mask
    
    # KNN→PCA→取最小特征向量（Open3D 的 estimate_normals 内部完成）
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    k_eff = max(3, min(k, N)) # 确保KNN邻居数>=3且<=N
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_eff))
    
    normals = np.asarray(pcd.normals, dtype=np.float32)
    
    is_bad_mask = (~np.isfinite(normals).all(axis=1)) | (np.linalg.norm(normals, axis=1) < 1e-12)
    bad_cnt = int(is_bad_mask.sum())
    if bad_cnt > 0:
        print(f"[estimate_normals] N={N}, k_eff={k_eff} -> bad_normals={bad_cnt}/{N} "
              f"({bad_cnt / N:.2%}), criterion: non-finite or ||n||< 1e-12")
    else:
        print(f"[estimate_normals] N={N}, k_eff={k_eff} -> bad_normals=0/{N}")
    
    # # 把非有限/几乎为零的法线替换为随机单位向量（如果想这样的话，就把下面三行代码取消注释）
    # r = np.random.randn(bad_cnt.sum(), 3).astype(np.float32)
    # r /= (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    # normals[is_bad_mask] = r
    
    
    return normals, is_bad_mask
