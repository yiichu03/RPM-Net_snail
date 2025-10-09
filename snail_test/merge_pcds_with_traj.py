#!/usr/bin/env python3
# merge_pcds_with_traj.py
# 功能: 读取 PCD 序列 + 轨迹(ref_tls_T_xt32.csv)，
#       1) 合并点云到同一地图坐标系 -> merged_full.pcd
#       2) 保存用到的位姿位置点 -> traj_points.pcd
# python merge_pcds_with_traj.py --pcd_dir eagleg7/pcl  --traj_csv ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_xt32.csv  --out_pcd merged_full.pcd  --traj_points_out traj_points.pcd

import argparse, os, glob
import numpy as np
import pandas as pd
import open3d as o3d

# ---- math ----
def quat_xyzw_to_R(q):
    q = np.asarray(q, float)
    q /= np.linalg.norm(q) + 1e-12
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)]
    ], float)

# ---- IO ----
def load_traj(csv_path):
    """从 ref_tls_T_xt32.csv 读取时间、位置、姿态(四元数: x y z w)"""
    df = pd.read_csv(csv_path)
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    pos_cols  = ['M_p_L_x','M_p_L_y','M_p_L_z']
    quat_cols = ['M_q_L_x','M_q_L_y','M_q_L_z','M_q_L_w']
    times = df[time_col].to_numpy(float)
    M_p_L   = df[pos_cols].to_numpy(float)
    M_q_L  = df[quat_cols].to_numpy(float)
    return times, M_p_L, M_q_L

def list_pcds(pcd_dir):
    files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not files:
        raise FileNotFoundError(f'未在 {pcd_dir} 找到 .pcd')
    # 默认从文件名解析时间戳（形如 1696641886.183674812.pcd）
    ts = []
    for f in files:
        try: ts.append(float(os.path.splitext(os.path.basename(f))[0]))
        except: ts.append(np.nan)
    ts = np.array(ts, float)
    if np.isnan(ts).any():
        bad = [files[i] for i in np.where(np.isnan(ts))[0][:5]]
        raise ValueError(f'有文件名不是时间戳，示例: {bad}')
    return files, ts

def nearest_indices(query, base):
    """对已排序时间轴 base，为每个 query 找最近邻nearest neighbor索引
    先用二分锁定左右邻，再比距离选更近的那个索引
    query: pcd_ts
    base: traj_t
    """
    # 二分：找到把 query 插入 base 后仍保持有序的位置
    # 语义：base[idx-1] <= query < base[idx]
    idx = np.searchsorted(base, query)
    # 处理边界，保证后面访问 idx-1 和 idx 不越界
    idx = np.clip(idx, 1, len(base)-1)
    # 最近邻只可能是左邻或右邻
    left, right = idx-1, idx
    # 比较到左右邻的距离；距离相等时偏向左（<=）
    choose_left = (np.abs(base[left]-query) <= np.abs(base[right]-query))
    # 按位选择：True 选 left，False 选 right
    idx_nn = np.where(choose_left, left, right)
    
    return idx_nn

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd_dir', required=True, help='PCD 序列目录 (pcl/ 或 enhanced/)')
    ap.add_argument('--traj_csv', required=True, help='ref_tls_T_xt32.csv 路径')
    ap.add_argument('--out_pcd', default='merged_full.pcd', help='合并点云输出文件名')
    ap.add_argument('--traj_points_out', default='traj_points.pcd', help='轨迹位置点输出文件名')
    ap.add_argument('--time_tolerance', type=float, default=0.05, help='允许的最近邻时间差(秒)')
    args = ap.parse_args()

    traj_t, traj_M_p_L, traj_M_q_L = load_traj(args.traj_csv) # times, pos位置, quat姿态
    pcd_files, pcd_ts = list_pcds(args.pcd_dir)

    # 最近邻时间配对
    nn_idx = nearest_indices(pcd_ts, traj_t)
    dt = np.abs(traj_t[nn_idx] - pcd_ts) # 轨迹的时间戳和pcd的时间戳的差

    all_xyz = []
    pose_pts = []
    used = 0

    for i, f in enumerate(pcd_files):
        if dt[i] > args.time_tolerance:
            print(f'[跳过] {os.path.basename(f)} 时间差 {dt[i]:.3f}s 超过容差')
            continue
        t = traj_M_p_L[nn_idx[i]]
        q = traj_M_q_L[nn_idx[i]]
        R = quat_xyzw_to_R(q)

        pc = o3d.io.read_point_cloud(f)
        if len(pc.points) == 0:
            continue
        xyz = np.asarray(pc.points, float) # 坐标
        xyz_M = (xyz @ R.T) + t # (N_i, 3)（N_i 为该帧点数）
        # list: 每个元素(N_i, 3)
        all_xyz.append(xyz_M)
        # list: 每个元素(3,)
        pose_pts.append(t) # 记录位置点
        used += 1
        if used % 200 == 0:
            print(f'已处理 {used} 帧...')

    if used == 0:
        raise RuntimeError('没有任何帧通过时间容差筛选，请检查时间或放宽 --time_tolerance')

    # 保存合并点云
    merged = o3d.geometry.PointCloud()
    # np.vstack(...) 把它们纵向拼成一个大 M×3 数组。
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
    o3d.io.write_point_cloud(args.out_pcd, merged)

    # 保存轨迹位置点
    pose_pts = np.vstack(pose_pts) # (used, 3)
    traj_pts = o3d.geometry.PointCloud()
    traj_pts.points = o3d.utility.Vector3dVector(pose_pts)
    traj_pts.paint_uniform_color([0.0, 0.0, 1.0]) # 蓝色
    o3d.io.write_point_cloud(args.traj_points_out, traj_pts)

    print(f'[完成] 使用 {used}/{len(pcd_files)} 帧')
    print(f'[输出] 合并点云 -> {args.out_pcd}  点数={np.asarray(merged.points).shape[0]}')
    print(f'[输出] 轨迹位置点 -> {args.traj_points_out}  点数={np.asarray(traj_pts.points).shape[0]}')

if __name__ == '__main__':
    main()
