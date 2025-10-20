# merge_pcds_with_traj.py
# 功能: 读取 PCD 序列 + 轨迹(ref_tls_T_xt32.csv)，
#       1) 合并点云到同一地图坐标系 -> merged_full.pcd
#       2) 保存用到的位姿位置点 -> traj_points.pcd
# python merge_pcds_with_traj.py --pcd_dir eagleg7/pcl  --traj_csv ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_oculii.csv  --out_pcd merged_full.pcd  --traj_points_out traj_points.pcd
# 弃用python merge_pcds_with_traj.py --pcd_dir eagleg7/pcl  --traj_csv ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_xt32.csv  --out_pcd merged_full_wrong.pcd  --traj_points_out traj_points_wrong.pcd

import argparse, os, glob
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import bisect
def R_from_q(q):
    '''
    由四元数得到旋转矩阵'''
    q = np.asarray(q, float)
    q /= np.linalg.norm(q) + 1e-12
    return R.from_quat(q).as_matrix()

def load_traj(csv_path):
    '''从ref_tls_T_oculii.csv读取时间、位置、姿态(四元数: x y z w)'''
    df = pd.read_csv(csv_path)
    times = df['time'].to_numpy(float)
    pos_cols  = ['M_p_O_x','M_p_O_y','M_p_O_z']
    quat_cols = ['M_q_O_x','M_q_O_y','M_q_O_z','M_q_O_w']
    M_p_O   = df[pos_cols].to_numpy(float)
    M_q_O  = df[quat_cols].to_numpy(float)
    return times, M_p_O, M_q_O

def list_pcds(pcd_dir):
    files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not files:
        raise FileNotFoundError(f'未在 {pcd_dir} 找到 .pcd')
    # 默认从文件名解析时间戳（形如 1696641886.183674812.pcd）
    ts = []
    for f in files:
        ts.append(float(os.path.splitext(os.path.basename(f))[0]))
    ts = np.array(ts, float)
    return files, ts

def nearest_indices(query, base):
    """ 二分查找 base 必须升序。用 bisect 在 O(log N) 找左右邻，再比谁更近。"""
    base = list(base)  # bisect 需要序列
    out = []
    for q in query:
        j = bisect.bisect_left(base, q)            # base[j-1] <= q < base[j]
        if j == 0: 
            out.append(0)
        elif j == len(base):
            out.append(len(base)-1)
        else:
            left, right = j-1, j
            # 距离相等时偏向左邻（<=）
            i = left if abs(base[left]-q) <= abs(base[right]-q) else right
            out.append(i)
    return np.array(out, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd_dir', required=True, help='PCD 序列目录 (pcl/ 或 enhanced/)')
    ap.add_argument('--traj_csv', required=True, help='ref_tls_T_oculii.csv 路径')
    ap.add_argument('--out_pcd', default='merged_full.pcd', help='合并点云输出文件名')
    ap.add_argument('--traj_points_out', default='traj_points.pcd', help='轨迹位置点输出文件名')
    ap.add_argument('--time_tolerance', type=float, default=0.05, help='允许的最近邻时间差(秒)')
    args = ap.parse_args()

    traj_t, traj_M_p_O, traj_M_q_O = load_traj(args.traj_csv) # times, pos位置, quat姿态
    pcd_files, pcd_ts = list_pcds(args.pcd_dir)

    # 最近邻时间配对
    idx_nn = nearest_indices(pcd_ts, traj_t)
    time_diffs = np.abs(pcd_ts - traj_t[idx_nn]) # 轨迹和pcd的时间戳的差值

    all_xyz = []  # 用于合并的点云
    pose_points = []  # 用于保存轨迹位置点
    used = 0

    for i, f in enumerate(pcd_files):
        if time_diffs[i] > args.time_tolerance:
            print(f'跳过 {os.path.basename(f)}，时间差 {time_diffs[i]:.3f}s 超过容限 {args.time_tolerance}s')
            continue
        p = traj_M_p_O[idx_nn[i]]
        q = traj_M_q_O[idx_nn[i]]
        w_R_o = R_from_q(q) 
        w_t_o = p           

        pc = o3d.io.read_point_cloud(f)
        if len(pc.points) == 0:
            continue
        xyz = np.asarray(pc.points, float) # 坐标
        xyz_cur = (xyz @ w_R_o.T) + w_t_o # 行向量 (N_i, 3)（N_i 为该帧点数）
        all_xyz.append(xyz_cur)
        pose_points.append(p) # 记录位置点
        used += 1
        if used % 200 == 0:
            print(f'已处理 {used} 帧...')
    
    if used == 0:
        print('未处理任何帧，退出')
        return
    
    # 保存合并点云
    merged_pc = o3d.geometry.PointCloud()
    # np.vstack(...) 把它们纵向拼成一个大 M×3 数组。
    merged_pc.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
    o3d.io.write_point_cloud(args.out_pcd, merged_pc)

    # 保存轨迹位置点
    traj_points = o3d.geometry.PointCloud()
    traj_points.points = o3d.utility.Vector3dVector(np.vstack(pose_points))
    traj_points.paint_uniform_color([0.0, 0.0, 1.0]) # 蓝色
    o3d.io.write_point_cloud(args.traj_points_out, traj_points)

    print(f'[完成] 使用 {used}/{len(pcd_files)} 帧')
    print(f'[输出] 合并点云 -> {args.out_pcd}  点数={np.asarray(merged_pc.points).shape[0]}')
    print(f'[输出] 轨迹位置点 -> {args.traj_points_out}  点数={np.asarray(traj_points.points).shape[0]}')

if __name__ == '__main__':
    main()

