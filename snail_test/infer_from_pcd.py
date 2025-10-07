from pathlib import Path
import sys

# infer_single_pair.py 位于 .../RPMNet/snail_test/
# 我们要把 .../RPMNet/src 加到 PYTHONPATH
SRC_DIR = Path(__file__).resolve().parents[1] / "src"   # parents[1] == RPMNet
sys.path.insert(0, str(SRC_DIR))

import argparse
import os

import numpy as np
import torch
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同级 tools.py / convert_radar_single_frames.py 稳定可见
from tools import save_inference_visuals, apply_se3
from convert_radar_single_frames import estimate_normals_for_radar
'''
# 选第0帧做 source，第1帧做 ref；自动估半径；保存可视化
python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5   --resume D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth   --src 0 --ref 3 --num_iter 10 --auto_radius --neighbors 50 --save_vis
python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5  --resume D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth  --src 0 --ref 3 --num_iter 10 --auto_radius --neighbors 50 --save_vis
'''
# --- utils -------------------------------------------------------------------

def to_tensor(x, device):
    return torch.from_numpy(x).float().to(device)


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
    """
    将单个PCD文件转换为点云数据和法向量。
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    # 估计法向量
    normals, is_bad_mask = estimate_normals_for_radar(
        points, k=20,
    )
    
    # 汇总帧内统计
    n_bad = int(is_bad_mask.sum())
    print("法向量无效的点数: {}/{}".format(n_bad, len(is_bad_mask)))

    xyz = np.asarray(points).astype(np.float32)
    normals = np.asarray(normals).astype(np.float32)

    normals = flip_normals_outward(xyz, normals)
    
    return xyz, normals

# --- main inference ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("RPMNet single-pair inference on SNAIL radar frames")
    parser.add_argument('--resume', type=str, required=True, help='RPMNet checkpoint .pth')
    parser.add_argument('--num_iter', type=int, default=5, help='RPMNet registration iterations')
    parser.add_argument('--auto_radius', action='store_true', help='estimate feature radius from data')
    parser.add_argument('--radius', type=float, default=0.3, help='feature radius (ignored if --auto_radius)')
    parser.add_argument('--neighbors', type=int, default=30, help='neighbors per point for features')
    parser.add_argument('--save_vis', action='store_true', help='save before/after PLY into same folder as H5')
    parser.add_argument('--src_pcd', type=str, required=True, help='source PCD path')
    parser.add_argument('--ref_pcd', type=str, required=True, help='reference PCD path')
    parser.add_argument('--out_dir', type=str, default=None, help='where to save outputs')
    
    args = parser.parse_args()

    out_dir = args.out_dir or str(Path(ref_pcd_path).parent)

    # device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("WARNING: using CPU for inference")

    src_pcd_path = args.src_pcd
    ref_pcd_path = args.ref_pcd

    assert Path(src_pcd_path).exists(), f"src_pcd not found: {src_pcd_path}"
    assert Path(ref_pcd_path).exists(), f"ref_pcd not found: {ref_pcd_path}"

    xyz_src, n_src = transfer_pcd_to_data(src_pcd_path)
    xyz_ref, n_ref = transfer_pcd_to_data(ref_pcd_path)
    print(f"  Source frame {src_pcd_path}: {len(xyz_src)} points")
    print(f"  Reference frame {ref_pcd_path}: {len(xyz_ref)} points")


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

    results = save_inference_visuals(
        out_dir=out_dir,
        transforms=transforms,
        endpoints=endpoints,
        xyz_src=xyz_src, xyz_ref=xyz_ref,
        n_src=n_src, n_ref=n_ref,
        src_idx=Path(src_pcd_path).stem, ref_idx=Path(ref_pcd_path).stem,
        save_ply=args.save_vis,
    )
    
if __name__ == '__main__':
    main()
