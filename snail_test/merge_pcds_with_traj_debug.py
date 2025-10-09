#!/usr/bin/env python3
# merge_with_traj_debug_v2.py
# 单窗口可视化；强显轨迹（蓝线/可选圆柱管）；红色路标球；灰色点云；时间对齐统计
# python merge_pcds_with_traj_debug.py --pcd_dir eagleg7/pcl --traj_csv ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_xt32.csv --out_pcd merged_full.pcd --no_merge --traj_tube_radius 0.08 --traj_mark_every 1

import argparse, os, glob, numpy as np, pandas as pd, open3d as o3d

# ---------- math utils ----------
def quat_xyzw_to_R(q):
    q = np.asarray(q, float); q /= np.linalg.norm(q)
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z; xy,xz,yz = x*y, x*z, y*z; wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)]
    ], float)

def quat_wxyz_to_R(q):
    w,x,y,z = q
    return quat_xyzw_to_R([x,y,z,w])

def transform_points(xyz, R, t):
    return (xyz @ R.T) + t

def rodrigues(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x,y,z = axis; c = np.cos(angle); s = np.sin(angle); C = 1-c
    return np.array([
        [c+x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c+y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c+z*z*C]
    ], float)

# ---------- IO ----------
def load_traj(csv_path):
    df = pd.read_csv(csv_path)
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    pos_cols  = ['M_p_L_x','M_p_L_y','M_p_L_z']
    quat_cols = ['M_q_L_x','M_q_L_y','M_q_L_z','M_q_L_w']
    times = df[time_col].to_numpy(float)
    pos   = df[pos_cols].to_numpy(float)
    quat  = df[quat_cols].to_numpy(float)
    return times, pos, quat

def list_pcds(pcd_dir, times_txt=None):
    files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not files: raise FileNotFoundError(f'未在 {pcd_dir} 找到 .pcd')
    if times_txt:
        ts = np.loadtxt(times_txt, dtype=float)
        n = min(len(ts), len(files))
        if n != len(files):
            print(f'[警告] times.txt 行数({len(ts)}) ≠ pcd 数量({len(files)})，按最短长度截断 -> {n}')
        return files[:n], ts[:n]
    # 默认从文件名解析时间戳
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
    idx = np.searchsorted(base, query)
    idx = np.clip(idx, 1, len(base)-1)
    left, right = idx-1, idx
    choose_left = (np.abs(base[left]-query) <= np.abs(base[right]-query))
    return np.where(choose_left, left, right)

# ---------- visualization helpers ----------
def make_traj_lineset(pts, color=(0,0,1)):
    if len(pts) < 2: return None
    lines = [[i, i+1] for i in range(len(pts)-1)]
    colors = [color for _ in lines]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def lines_to_tubes(pts, radius=0.05, color=(0,0,1)):
    """把折线段转成圆柱管（粗轨迹），便于“看得见”"""
    if len(pts) < 2: return []
    meshes = []
    z_axis = np.array([0,0,1.0])
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        v = p1 - p0
        L = np.linalg.norm(v)
        if L < 1e-6: continue
        dir = v / L
        # 旋转: z轴 -> dir
        dot = float(np.clip(np.dot(z_axis, dir), -1.0, 1.0))
        if abs(dot-1.0) < 1e-6:
            R = np.eye(3)
        elif abs(dot+1.0) < 1e-6:
            R = rodrigues(np.array([1,0,0]), np.pi)
        else:
            axis = np.cross(z_axis, dir)
            angle = np.arccos(dot)
            R = rodrigues(axis, angle)
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(color)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = (p0 + p1)/2.0
        cyl.transform(T)
        meshes.append(cyl)
    return meshes

def coord_frame(T, size=1.0):
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cf.transform(T); return cf

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd_dir', required=True)
    ap.add_argument('--traj_csv', required=True)
    ap.add_argument('--out_pcd', default='merged_map.pcd')
    ap.add_argument('--times_txt', default=None)
    ap.add_argument('--time_tolerance', type=float, default=0.05)
    ap.add_argument('--quat_format', choices=['xyzw','wxyz'], default='xyzw')
    ap.add_argument('--invert_rot', action='store_true')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--no_merge', action='store_true')
    ap.add_argument('--traj_mark_every', type=int, default=10, help='每 N 帧放一个红色路标球')
    ap.add_argument('--traj_tube_radius', type=float, default=0.0, help='>0 则用圆柱管显示轨迹粗线')
    args = ap.parse_args()

    traj_t, traj_p, traj_q = load_traj(args.traj_csv)
    pcd_files, pcd_ts = list_pcds(args.pcd_dir, args.times_txt)

    makeR = quat_xyzw_to_R if args.quat_format == 'xyzw' else quat_wxyz_to_R

    used_xyz, used_pose_pts, used_Ts, used_dt = [], [], [], []
    nn_idx = nearest_indices(pcd_ts, traj_t)
    dt = np.abs(traj_t[nn_idx] - pcd_ts)

    used_cnt = 0
    for i, f in enumerate(pcd_files):
        if dt[i] > args.time_tolerance: continue
        p = traj_p[nn_idx[i]]
        q = traj_q[nn_idx[i]]
        R = makeR(q)
        if args.invert_rot: R = R.T

        if not args.no_merge:
            pc = o3d.io.read_point_cloud(f)
            if len(pc.points) == 0: continue
            xyz = np.asarray(pc.points, float)
            xyz_M = transform_points(xyz, R, p)
            used_xyz.append(xyz_M)

        used_pose_pts.append(p)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = p
        used_Ts.append(T)
        used_dt.append(dt[i])
        used_cnt += 1
        if args.max_frames and used_cnt >= args.max_frames: break

    if used_cnt == 0:
        raise RuntimeError('没有任何帧通过时间容差筛选，请放宽 --time_tolerance 或检查时间来源/顺序。')

    used_dt = np.array(used_dt)
    print(f'[统计] 使用了 {used_cnt}/{len(pcd_files)} 帧')
    print(f'[时间差] mean={used_dt.mean():.4f}s  max={used_dt.max():.4f}s  95%={np.percentile(used_dt,95):.4f}s')

    # ---- 组装可视化几何 ----
    geoms = []

    # 合并点云 + 统一灰色，避免遮挡轨迹颜色
    if not args.no_merge:
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack(used_xyz))
        merged.paint_uniform_color([0.7,0.7,0.7])
        o3d.io.write_point_cloud(args.out_pcd, merged, write_ascii=False, compressed=True)
        print(f'[输出] merged map -> {args.out_pcd}  点数={np.asarray(merged.points).shape[0]}')
        geoms.append(merged)

    # 轨迹（蓝线 or 圆柱管）
    pose_pts = np.vstack(used_pose_pts)
    # === 保存轨迹到文件（便于 CloudCompare 叠加）===
    # 1) 轨迹折线（LineSet） -> PLY
    traj_ls = make_traj_lineset(pose_pts, color=(0,0,1))
    if traj_ls is not None:
        o3d.io.write_line_set('traj_used.ply', traj_ls)
        print('[输出] traj_used.ply (LineSet)')

    # 2) 轨迹点（每帧位姿位置） -> PCD
    traj_pts = o3d.geometry.PointCloud()
    traj_pts.points = o3d.utility.Vector3dVector(pose_pts)
    traj_pts.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    o3d.io.write_point_cloud('traj_points.pcd', traj_pts, write_ascii=False, compressed=True)
    print('[输出] traj_points.pcd (PointCloud)')

    # 3) 如果你使用了圆柱“粗管线”，也可顺便导出成一个三角网格 PLY（可选）
    if args.traj_tube_radius > 0:
        tubes_meshes = lines_to_tubes(pose_pts, radius=args.traj_tube_radius, color=(0,0,1))
        if tubes_meshes:
            merged_mesh = o3d.geometry.TriangleMesh()
            for m in tubes_meshes:
                merged_mesh += m
            merged_mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh('traj_tubes.ply', merged_mesh, write_ascii=False)
            print('[输出] traj_tubes.ply (TriangleMesh)')

    if args.traj_tube_radius > 0:
        geoms.extend(lines_to_tubes(pose_pts, radius=args.traj_tube_radius, color=(0,0,1)))
    else:
        traj_ls = make_traj_lineset(pose_pts, color=(0,0,1))
        if traj_ls: geoms.append(traj_ls)

    # 红色路标球
    N, r = max(1, args.traj_mark_every), 0.3
    for i in range(0, len(used_Ts), N):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.paint_uniform_color([1,0,0])
        s.translate(used_Ts[i][:3,3])
        geoms.append(s)

    # 首/中/末坐标系，放大一点更显眼
    idx_show = [0, len(used_Ts)//2, len(used_Ts)-1] if len(used_Ts) >= 3 else [0, len(used_Ts)-1]
    for k in sorted(set(idx_show)):
        geoms.append(coord_frame(used_Ts[k], size=1.0))

    # ---- 单窗口渲染 ----
    vis = o3d.visualization.Visualizer()
    vis.create_window('Merged + Trajectory Debug (single window)')
    for g in geoms: vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0,0,0])  # 黑底
    opt.line_width = 6.0
    opt.point_size = 1.5
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
