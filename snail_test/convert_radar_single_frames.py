# convert_radar_single_frames.py
import numpy as np
import h5py
import os
from pathlib import Path
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

def write_log(log_path: str, msg: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def convert_single_radar_frames(
    radar_dir: str,
    output_dir: str,
    frame_indices: list = [0, 10, 20],  # 选择第0、10、20帧
    downsample_points: int = 1024,
    k_normal: int = 20,
    log_filename: str = "normals_log.txt"
):
    """
    转换选定的雷达帧为RPM-Net格式，并记录法向量估计日志
    """
    print("=== 步骤1: 数据转换 ===")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    # 日志头
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] 启动转换\n")
        f.write(f"radar_dir={radar_dir}, output_dir={output_dir}, k_normal={k_normal}, downsample_points={downsample_points}\n")

    # 1. 读取所有PCD文件
    pcd_files = sorted(list(Path(radar_dir).glob("*.pcd")))
    print(Path(radar_dir).resolve())
    print(f"找到 {len(pcd_files)} 个PCD文件")
    write_log(log_path, f"找到 {len(pcd_files)} 个PCD文件")

    if len(pcd_files) == 0:
        raise ValueError("未找到PCD文件！")

    # 2. 选择指定帧
    selected_files = [pcd_files[i] for i in frame_indices if i < len(pcd_files)]
    chosen_names = [pcd_files[i].name for i in frame_indices if i < len(pcd_files)]
    print(f"选择帧: {chosen_names}")
    write_log(log_path, f"选择帧: {chosen_names}")

    # 3. 处理选定的帧
    all_point_clouds = []
    all_normals = []
    all_labels = []
    all_random_masks = []  # 帧内每个点是否随机法向量
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

        # 下采样（不足则重复）
        if len(points) >= downsample_points:
            indices = np.random.choice(len(points), downsample_points, replace=False)
            sampled_points = points[indices]
            write_log(log_path, f"  随机下采样 {downsample_points} 点（从 {len(points)} 中）")
        else:
            repeat_times = downsample_points // len(points)
            remainder = downsample_points % len(points)
            sampled_points = np.tile(points, (repeat_times, 1))
            if remainder > 0:
                sampled_points = np.vstack([sampled_points, points[:remainder]])
            write_log(log_path, f"  点不足，重复采样到 {downsample_points} 点（原始 {len(points)}）")

        print(f"  下采样后: {len(sampled_points)} 点")

        # 估计法向量（带日志 & 随机掩码）
        normals, is_random_mask = estimate_normals_for_radar(
            sampled_points, k=k_normal,
            log_path=log_path,
            context=f"frame_idx={i}, file={pcd_file.name}"
        )

        # 汇总帧内统计
        n_rand = int(is_random_mask.sum())
        write_log(log_path, f"  帧[{i}] 随机法向量点数: {n_rand}/{len(is_random_mask)}")

        all_point_clouds.append(sampled_points)
        all_normals.append(normals)
        all_random_masks.append(is_random_mask.astype(np.uint8))
        all_labels.append(i)  # 用帧索引作为标签
        # 文件名是时间戳（如 1706001766.376780611），安全转为浮点
        try:
            timestamps.append(float(pcd_file.stem))
        except Exception:
            timestamps.append(float(i))

    # 4. 转换为数组格式
    point_clouds_array = np.array(all_point_clouds)        # (N_frames, 1024, 3)
    normals_array = np.array(all_normals)                  # (N_frames, 1024, 3)
    labels_array = np.array(all_labels)                    # (N_frames,)
    random_mask_array = np.array(all_random_masks)         # (N_frames, 1024)

    print(f"最终数据形状: points={point_clouds_array.shape}, normals={normals_array.shape}, random_mask={random_mask_array.shape}")
    write_log(log_path, f"最终数据形状: points={point_clouds_array.shape}, normals={normals_array.shape}, random_mask={random_mask_array.shape}")

    # 5. 保存HDF5文件
    h5_file = os.path.join(output_dir, 'radar_single_frames_test0.h5')
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('data', data=point_clouds_array)
        f.create_dataset('normal', data=normals_array)
        f.create_dataset('label', data=labels_array)
        f.create_dataset('normal_is_random', data=random_mask_array)  # 新增：法向量是否随机

    print(f"HDF5文件已保存: {h5_file}")
    write_log(log_path, f"HDF5文件已保存: {h5_file}")

    # 6. 创建元数据文件
    with open(os.path.join(output_dir, 'test_files.txt'), 'w', encoding='utf-8') as f:
        f.write("radar_single_frames/radar_single_frames_test0.h5\n")

    with open(os.path.join(output_dir, 'shape_names.txt'), 'w', encoding='utf-8') as f:
        f.write("radar_frame\n")

    # 7. 保存时间戳与随机掩码（npy）
    np.save(os.path.join(output_dir, 'timestamps.npy'), np.array(timestamps))
    np.save(os.path.join(output_dir, 'normal_is_random.npy'), random_mask_array)

    write_log(log_path, "=== 数据转换完成 ===")
    print("=== 数据转换完成 ===")
    return point_clouds_array, normals_array, labels_array, random_mask_array

def estimate_normals_for_radar(points: np.ndarray, k: int = 20, log_path: str = None, context: str = ""):
    """
    为雷达数据估计法向量。
    返回:
        normals: (N, 3) 归一化法向量
        is_random_mask: (N,) 布尔数组，True 表示该点法向量是随机生成的
    日志：
        - points数量、k
        - 是否整体随机（点数不足 / 异常）
        - PCA成功时，给出若干质量统计（如低表面变化比例）
    """
    N = len(points)
    if log_path:
        write_log(log_path, f"[estimate] {context} | N={N}, k={k}")

    # 情况1：整体点数不足 -> 全部随机
    if N < k:
        normals = np.random.randn(N, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        is_random_mask = np.ones(N, dtype=bool)
        if log_path:
            write_log(log_path, f"[estimate] {context} | N<k，全部使用随机法向量")
        return normals, is_random_mask

    try:
        # 使用KNN估计法向量
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        _, indices = nbrs.kneighbors(points)

        normals = np.zeros((N, 3), dtype=np.float64)
        is_random_mask = np.zeros(N, dtype=bool)

        # 可选的质量指标（表面变化 surface variation）
        # sv = λ_min / (λ1+λ2+λ3)，值越小说明越像平面
        surface_variations = np.zeros(N, dtype=np.float64)

        for i in range(N):
            neighbor_points = points[indices[i]]
            centered = neighbor_points - np.mean(neighbor_points, axis=0)

            # 理论上 k>=3，这里做一次防御性判断
            if centered.shape[0] < 3:
                n = np.random.randn(3)
                n /= (np.linalg.norm(n) + 1e-12)
                normals[i] = n
                is_random_mask[i] = True
                continue

            cov = np.cov(centered.T)
            # 防止极端退化导致NaN
            if not np.all(np.isfinite(cov)):
                n = np.random.randn(3)
                n /= (np.linalg.norm(n) + 1e-12)
                normals[i] = n
                is_random_mask[i] = True
                continue

            evals, evecs = np.linalg.eigh(cov)
            # 排序（从小到大）
            evals = np.clip(np.sort(evals), a_min=0.0, a_max=None)
            n = evecs[:, 0]  # 最小特征值对应的特征向量
            n /= (np.linalg.norm(n) + 1e-12)

            normals[i] = n
            is_random_mask[i] = False

            denom = evals.sum() + 1e-12
            surface_variations[i] = evals[0] / denom

        if log_path:
            pct_random = 100.0 * is_random_mask.mean()
            sv = surface_variations[np.isfinite(surface_variations)]
            sv_med = float(np.median(sv)) if sv.size else float('nan')
            sv_p10 = float(np.percentile(sv, 10)) if sv.size else float('nan')
            sv_p90 = float(np.percentile(sv, 90)) if sv.size else float('nan')
            low_sv_ratio = float(np.mean(sv < 1e-4)) if sv.size else float('nan')  # 非常平坦的邻域比例
            write_log(log_path, f"[estimate] {context} | PCA成功: 随机比例={pct_random:.2f}%, "
                                f"SV[med={sv_med:.2e}, p10={sv_p10:.2e}, p90={sv_p90:.2e}], "
                                f"SV<1e-4比例={low_sv_ratio:.2f}")

        return normals, is_random_mask

    except Exception as e:
        # 情况2：发生异常 -> 全部随机
        normals = np.random.randn(N, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        is_random_mask = np.ones(N, dtype=bool)
        if log_path:
            write_log(log_path, f"[estimate] {context} | 发生异常，全部使用随机法向量: {repr(e)}")
        return normals, is_random_mask


# 运行转换
if __name__ == "__main__":
    convert_single_radar_frames( 
        radar_dir="./eagleg7/enhanced/",
        output_dir="radar_single_frames/",
        frame_indices=[0, 10, 20],  # 选择3帧测试
        downsample_points=1024,
        k_normal=30,
        log_filename="normals_log.txt"
    )
