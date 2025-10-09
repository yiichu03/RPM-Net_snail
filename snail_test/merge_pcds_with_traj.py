#!/usr/bin/env python3
# merge_pcds_with_traj.py
# 功能: 用 ref_tls_T_xt32.csv 的 LiDAR位姿把 pcd 序列变换到同一坐标系, 合并后保存/查看
# python merge_pcds_with_traj.py --pcd_dir eagleg7/pcl --traj_csv ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_xt32.csv --out_pcd merged_full.pcd

import argparse, os, glob
import numpy as np
import pandas as pd
import open3d as o3d

def quat_xyzw_to_R(q):
    # q: [x,y,z,w]
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)]
    ], dtype=float)

def load_traj(csv_path):
    df = pd.read_csv(csv_path)
    # 容错：自动找出时间列名（包含'time'），以及位姿列
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    pos_cols  = ['M_p_L_x','M_p_L_y','M_p_L_z']
    quat_cols = ['M_q_L_x','M_q_L_y','M_q_L_z','M_q_L_w']
    times = df[time_col].to_numpy(dtype=float)
    pos   = df[pos_cols].to_numpy(dtype=float)
    quat  = df[quat_cols].to_numpy(dtype=float)
    return times, pos, quat

def nearest_pose(t, traj_times, pos, quat):
    i = np.argmin(np.abs(traj_times - t))
    return pos[i], quat[i], float(traj_times[i])

def transform_points_xyz(xyz, R, t):
    return (xyz @ R.T) + t  # Nx3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd_dir', required=True, help='PCD序列目录（pcl/ 或 enhanced/）')
    ap.add_argument('--traj_csv', required=True, help='ref_tls_T_xt32.csv 路径')
    ap.add_argument('--out_pcd', default='merged_map.pcd', help='合并后输出路径')
    ap.add_argument('--voxel', type=float, default=0.0, help='体素降采样尺寸(米)，0关闭')
    ap.add_argument('--stride', type=int, default=1, help='抽帧间隔。例如 5 表示每5帧取1帧')
    ap.add_argument('--max_frames', type=int, default=0, help='最多处理的帧数，0为不限')
    ap.add_argument('--time_tolerance', type=float, default=0.05, help='时间最近邻容差(秒)，超出则跳过该帧')
    args = ap.parse_args()

    traj_t, traj_p, traj_q = load_traj(args.traj_csv)

    pcd_files = sorted(glob.glob(os.path.join(args.pcd_dir, '*.pcd')))
    if not pcd_files:
        raise FileNotFoundError(f'在 {args.pcd_dir} 未找到 .pcd')

    all_xyz = []
    used = 0
    for idx, f in enumerate(pcd_files[::args.stride]):
        # 文件名即时间戳（形如 1696641886.183674812.pcd）
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            t = float(stem)
        except:
            # 若文件名不是时间戳，可自行改为读取 times.txt；这里保持简洁
            print(f'[跳过] 无法解析时间戳: {f}')
            continue

        p, q, t_near = nearest_pose(t, traj_t, traj_p, traj_q)
        if abs(t_near - t) > args.time_tolerance:
            print(f'[跳过] {stem} 最近邻轨迹时间差 {abs(t_near - t):.3f}s 超过容差')
            continue

        pc = o3d.io.read_point_cloud(f)
        if len(pc.points) == 0: 
            continue
        xyz = np.asarray(pc.points, dtype=float)

        R = quat_xyzw_to_R(q)
        xyz_M = transform_points_xyz(xyz, R, p)  # 直接从LiDAR->地图M

        all_xyz.append(xyz_M)
        used += 1
        if args.max_frames and used >= args.max_frames:
            break
        if used % 20 == 0:
            print(f'已合并 {used} 帧...')

    if not all_xyz:
        raise RuntimeError('没有可合并的点（可能时间未对齐或容差太严）。')

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))

    if args.voxel > 0:
        merged = merged.voxel_down_sample(args.voxel)

    o3d.io.write_point_cloud(args.out_pcd, merged, write_ascii=False, compressed=True)
    print(f'完成: 合并 {used} 帧 -> {args.out_pcd}，点数={np.asarray(merged.points).shape[0]}')

    # 可视化（可注释掉）
    o3d.visualization.draw_geometries([merged], window_name='Merged Map')

if __name__ == '__main__':
    main()
