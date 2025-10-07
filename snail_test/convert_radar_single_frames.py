# convert_radar_single_frames.py
import numpy as np
import h5py
import os
from pathlib import Path
import open3d as o3d
from datetime import datetime

def write_log(log_path: str, msg: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def convert_single_radar_frames(
    radar_dir: str,
    output_dir: str,
    frame_indices: list = [0, 10, 20],  # 选择第0、10、20帧
    k_normal: int = 20,
    log_filename: str = "normals_log.txt",
):
    """
    转换选定的雷达帧为RPM-Net格式，并记录法向量估计日志
    
    Args:
        radar_dir: 输入PCD文件目录
        output_dir: 输出HDF5文件目录
        frame_indices: 要转换的帧索引列表
        k_normal: 法向量估计的KNN邻居数
        log_filename: 日志文件名
    """
    print("=== 步骤1: 数据转换 ===")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    # 日志头
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] 启动转换\n")
        f.write(f"radar_dir={radar_dir}, output_dir={output_dir}, k_normal={k_normal}\n")

    # 1. 读取所有PCD文件
    pcd_files = sorted(Path(radar_dir).glob("*.pcd"), key=lambda p: float(p.stem))
    print(Path(radar_dir).resolve())
    print(f"找到 {len(pcd_files)} 个PCD文件")
    write_log(log_path, f"找到 {len(pcd_files)} 个PCD文件")

    if len(pcd_files) == 0:
        raise ValueError("未找到PCD文件！")

    # 2. 选择指定帧
    selected_idx   = [i for i in frame_indices if i < len(pcd_files)]
    if not selected_idx:
        raise ValueError(f"frame_indices 全部越界：有效范围是 [0, {len(pcd_files)-1}]")
    selected_files = [pcd_files[i] for i in frame_indices if i < len(pcd_files)]
    chosen_names = [pcd_files[i].name for i in frame_indices if i < len(pcd_files)]
    print(f"选择帧: {chosen_names}")
    write_log(log_path, f"选择帧: {chosen_names}")
    
    # 保存frame_indices到文件名的映射
    frame_mapping_path = os.path.join(output_dir, 'frame_indices_mapping.txt')
    with open(frame_mapping_path, 'w', encoding='utf-8') as f:
        f.write("# Frame Index -> PCD Filename Mapping\n")
        f.write(f"# Total frames selected: {len(selected_files)}\n")
        f.write(f"# Original frame indices: {frame_indices}\n\n")
        for original_idx, pcd_file in zip(selected_idx, selected_files):
            f.write(f"{original_idx}\t{pcd_file.name}\n")
    print(f"Frame mapping saved to: {frame_mapping_path}")

    # 3. 处理选定的帧
    all_point_clouds = []
    all_normals = []
    all_labels = []
    timestamps = []

    for i, pcd_file in enumerate(selected_files):
        print(f"处理帧 {i+1}/{len(selected_files)}: {pcd_file.name}")
        write_log(log_path, f"—— 处理帧[{i}] 文件: {pcd_file.name}")

        # 读取PCD文件
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        points = np.asarray(pcd.points)

        if len(points) == 0:
            warn_msg = f"警告: {pcd_file.name} 是空的！跳过。"
            print(warn_msg)
            write_log(log_path, warn_msg)
            continue

        print(f"  原始点数: {len(points)}")
        write_log(log_path, f"  原始点数: {len(points)}")

        # 保留原始点数
        sampled_points = points
        
        # 估计法向量
        normals, is_bad_mask = estimate_normals_for_radar(
            sampled_points, k=k_normal,
        )

        # 汇总帧内统计
        n_bad = int(is_bad_mask.sum())
        write_log(log_path, f"  帧[{i}] 法向量无效的点数: {n_bad}/{len(is_bad_mask)}")

        all_point_clouds.append(sampled_points)
        all_normals.append(normals)
        all_labels.append(i)  # 用帧索引作为标签
        # 文件名是时间戳（如 1706001766.376780611），安全转为浮点
        try:
            timestamps.append(float(pcd_file.stem))
        except Exception:
            timestamps.append(float(i))

    # 4. 转换为数组格式
    # 注意：每帧点数可能不同，需要用object数组
    # 点数可能不同，使用object数组或者单独处理
    point_clouds_array = np.array(all_point_clouds, dtype=object)
    normals_array = np.array(all_normals, dtype=object)
    labels_array = np.array(all_labels)
    print(f"最终数据形状: points={point_clouds_array.shape}, normals={normals_array.shape}")
    write_log(log_path, f"最终数据形状: points={point_clouds_array.shape}, normals={normals_array.shape}")

    # 5. 保存HDF5文件
    h5_file = os.path.join(output_dir, 'radar_single_frames_test0.h5')
    with h5py.File(h5_file, 'w') as f:
        # 点数不同时，每个帧单独存储为一个dataset
        for idx, (pts, norms) in enumerate(zip(point_clouds_array, normals_array)):
            f.create_dataset(f'data_{idx}', data=pts)
            f.create_dataset(f'normal_{idx}', data=norms)
        # 存储元信息
        f.attrs['num_frames'] = len(point_clouds_array)
        f.attrs['point_counts'] = [len(pts) for pts in point_clouds_array]

        
        f.create_dataset('label', data=labels_array)

    print(f"HDF5文件已保存: {h5_file}")
    write_log(log_path, f"HDF5文件已保存: {h5_file}")

    # 6. 创建元数据文件
    with open(os.path.join(output_dir, 'test_files.txt'), 'w', encoding='utf-8') as f:
        f.write(str(Path(h5_file).resolve()) + "\n")

    with open(os.path.join(output_dir, 'shape_names.txt'), 'w', encoding='utf-8') as f:
        f.write("radar_frame\n")

    # 7. 保存时间戳
    np.save(os.path.join(output_dir, 'timestamps.npy'), np.array(timestamps))

    write_log(log_path, "=== 数据转换完成 ===")
    print("=== 数据转换完成 ===")
    return point_clouds_array, normals_array, labels_array

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



# 运行转换
if __name__ == "__main__":
    
    # 示例2: 保留原始点数
    convert_single_radar_frames( 
        radar_dir="./eagleg7/enhanced/",
        output_dir="radar_single_frames_original/",
        frame_indices=[0, 10, 20, 50],
        k_normal=30,
        log_filename="normals_log.txt",
    )
