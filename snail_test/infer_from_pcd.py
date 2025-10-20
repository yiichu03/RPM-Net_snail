from pathlib import Path
import sys
import pandas as pd
# 把 .../RPMNet/src 加到 PYTHONPATH
SRC_DIR = Path(__file__).resolve().parents[1] / "src"   # parents[1] == RPMNet
sys.path.insert(0, str(SRC_DIR))

import argparse
import os

import numpy as np
import torch
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同级 tools.py / convert_radar_single_frames.py 稳定可见
from tools import save_inference_visuals, estimate_normals_for_radar, write_hyperparams_txt, auto_radius
from merge_pcds_with_traj import R_from_q, load_traj, nearest_indices
# --- utils -------------------------------------------------------------------
def to_tensor(x, device):
    return torch.from_numpy(x).float().to(device)

def compose_ref_T_src(p_ref, q_ref, p_src, q_src):
    """给出世界系位姿: M_T_L_ref 与 M_T_L_src，返回 ref_T_src(R,t)"""
    R_ref = R_from_q(q_ref)
    R_src = R_from_q(q_src)
    t_ref = np.asarray(p_ref, float)
    t_src = np.asarray(p_src, float)
    R = R_ref.T @ R_src
    t = R_ref.T @ (t_src - t_ref)
    return R, t

def save_gt_txt(out_path, R, t, meta: dict):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# src_ts={meta['src_ts']:.9f}, ref_ts={meta['ref_ts']:.9f}\n")
        f.write(f"# src_nn_idx={meta['src_idx']}, ref_nn_idx={meta['ref_idx']}\n")
        f.write(f"# |src_dt|={meta['src_dt']:.6f}s, |ref_dt|={meta['ref_dt']:.6f}s\n")
        f.write("R=\n")
        for i in range(3):
            f.write("{:.9f} {:.9f} {:.9f}\n".format(*R[i]))
        f.write("t=\n{:.9f} {:.9f} {:.9f}\n".format(*t))
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t
        f.write("T_4x4=\n")
        for i in range(4):
            f.write("{:.9f} {:.9f} {:.9f} {:.9f}\n".format(*T[i]))

def flip_normals_outward(xyz, normals, origin=(0.0, 0.0, 0.0)):
    '''
    统一法线朝向
    因为很多法线估计（尤其是 PCA）会有符号不确定性
    PPF（Point Pair Feature）等法线相关特征对法线方向敏感
    '''
    o = np.asarray(origin, dtype=np.float32)[None, :] # (1,3) 参考点
    v = xyz - o                                       # (N,3) 每个点指向“外”的向量（点 - 参考点）
    flip = (np.sum(v * normals, axis=1) < 0)          # 点到外的向量 v 与法线 n 的点积 < 0 说明“朝里”
    normals[flip] = -normals[flip]                    # 把朝里的法线翻转
    return normals

def transfer_pcd_to_data(pcd_path):
    '''
    将单个pcd文件转换为点云数据和法向量'''
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 估计法线
    normals, is_bad_mask = estimate_normals_for_radar(points, k=20,)

    # 汇总统计
    n_bad = int(is_bad_mask.sum())
    print("法向量无效的点数: {}/{}".format(n_bad, len(is_bad_mask)))

    xyz = np.asarray(points).astype(np.float32)
    normals = np.asarray(normals).astype(np.float32)

    normals = flip_normals_outward(xyz, normals)
    
    return xyz, normals


def main():
    parser = argparse.ArgumentParser("RPMNet inference")
    parser.add_argument('--resume', type=str, required=True, help='RPMNet checkpoint .pth')
    parser.add_argument('--num_iter', type=int, default=5, help='RPMNet registration iterations')
    parser.add_argument('--auto_radius', action='store_true', help='estimate feature radius from data')
    parser.add_argument('--radius', type=float, default=0.3, help='feature radius (ignored if --auto_radius)')
    parser.add_argument('--neighbors', type=int, default=30, help='neighbors per point for features')
    parser.add_argument('--save_vis', action='store_true', help='save before/after PLY')
    parser.add_argument('--src_pcd', type=str, required=True, help='source PCD path')
    parser.add_argument('--ref_pcd', type=str, required=True, help='reference PCD path')
    parser.add_argument('--out_dir', type=str, default=None, help='where to save outputs')
    parser.add_argument('--traj_csv', type=str, required=True,
                    help='ref_tls_T_oculii.csv 路径')
    parser.add_argument('--time_tolerance', type=float, default=0.05,
                    help='pcd 与轨迹最近邻允许的时间差(秒)')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("WARNING: using CPU for inference")

    src_pcd_path = args.src_pcd
    ref_pcd_path = args.ref_pcd

    assert Path(src_pcd_path).exists(), f"src_pcd not found: {src_pcd_path}"
    assert Path(ref_pcd_path).exists(), f"ref_pcd not found: {ref_pcd_path}"

    out_dir = args.out_dir or str(Path(ref_pcd_path).parent)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 载入两帧 pcd，计算法向量
    xyz_src, n_src = transfer_pcd_to_data(src_pcd_path)
    xyz_ref, n_ref = transfer_pcd_to_data(ref_pcd_path)
    traj_t, traj_M_p_O, traj_M_q_O = load_traj(args.traj_csv)
    print(f"  Source frame {src_pcd_path}: {len(xyz_src)} points")
    print(f"  Reference frame {ref_pcd_path}: {len(xyz_ref)} points")

    # 2) 解析两帧 pcd 的时间戳
    ts_src = float(Path(src_pcd_path).stem)
    ts_ref = float(Path(ref_pcd_path).stem)

    # build model args namespace (keep in sync with your RPMNet implementation)
    # Using EarlyFusion defaults: features include xyz, dxyz and ppf
    if args.auto_radius:
        est_r = auto_radius(xyz_src, xyz_ref)
        print(f"[auto] feature radius ~= {est_r:.3f}")
        feat_radius = est_r
    else:
        feat_radius = args.radius

    from arguments import rpmnet_eval_arguments  # 或者 rpmnet_arguments，视仓库而定
    parser = rpmnet_eval_arguments()
    defaults = parser.parse_args([])   # 空列表 → 只拿默认值

    # 想改的参数
    defaults.radius        = feat_radius if args.auto_radius else args.radius
    defaults.num_neighbors = args.neighbors
    defaults.num_reg_iter  = args.num_iter
    defaults.no_slack      = False   

    final_radius = float(defaults.radius)

    write_hyperparams_txt(
        out_dir=out_dir,
        args=args,
        final_radius=final_radius,
        auto_radius_value=(est_r if args.auto_radius else None),
        defaults=defaults,
        device=str(device),
        n_src_pts=len(xyz_src),
        n_ref_pts=len(xyz_ref),
        auto_k=128, auto_pct=80,  # 与 auto_radius() 的默认一致
        normals_k=20              # 与 estimate_normals_for_radar(..., k=20) 一致
    )

    # import your project modules (assumes this script is launched from project root)
    import models.rpmnet as rpmnet_mod

    model = rpmnet_mod.get_model(defaults)
    model.to(device)
    model.eval()

    state = torch.load(args.resume, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)

    # assemble data dict expected by RPMNet forward
    pts_src6 = np.concatenate([xyz_src, n_src], axis=-1)[None, ...]   # (1, 1024, 6)
    pts_ref6 = np.concatenate([xyz_ref, n_ref], axis=-1)[None, ...]   # (1, 1024, 6)

    batch = {
        'points_src': to_tensor(pts_src6, device),
        'points_ref': to_tensor(pts_ref6, device)
    }

    with torch.no_grad():
        transforms, endpoints = model(batch, args.num_iter)

    # 3) 为各自时间找最近轨迹行
    nn_src = nearest_indices(np.array([ts_src]), traj_t)[0]
    nn_ref = nearest_indices(np.array([ts_ref]), traj_t)[0]
    dt_src = abs(traj_t[nn_src] - ts_src)
    dt_ref = abs(traj_t[nn_ref] - ts_ref)
    if (dt_src > args.time_tolerance) or (dt_ref > args.time_tolerance):
        print("[WARN] 轨迹时间差超过容差: src_dt={:.3f}s ref_dt={:.3f}s (容差 {:.3f}s)".format(
            float(dt_src), float(dt_ref), args.time_tolerance))

    # 4) 取两帧世界系位姿并计算 ref_T_src
    p_src = traj_M_p_O[nn_src]; q_src = traj_M_q_O[nn_src]
    p_ref = traj_M_p_O[nn_ref]; q_ref = traj_M_q_O[nn_ref]
    R_gt, t_gt = compose_ref_T_src(p_ref, q_ref, p_src, q_src)

    # 5) 存 GT 文本
    gt_txt = Path(out_dir) / "gt_ref_T_src.txt"
    save_gt_txt(str(gt_txt), R_gt, t_gt, meta={
        'src_ts': ts_src, 'ref_ts': ts_ref,
        'src_idx': int(nn_src), 'ref_idx': int(nn_ref),
        'src_dt': float(dt_src), 'ref_dt': float(dt_ref)
    })
    print(f"[GT] 写入 ground-truth 相对位姿 -> {gt_txt}")

    results = save_inference_visuals(
        out_dir=out_dir,
        transforms=transforms,
        endpoints=endpoints,
        xyz_src=xyz_src, xyz_ref=xyz_ref,
        n_src=n_src, n_ref=n_ref,
        src_idx=Path(src_pcd_path).stem, ref_idx=Path(ref_pcd_path).stem,
        save_ply=args.save_vis,
        gt_transform=(R_gt, t_gt) 
    )


    
if __name__ == '__main__':
    main()



