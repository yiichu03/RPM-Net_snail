"""创建合成测试数据 - 从同一个PCD文件生成source和target

这个脚本可以：
1. 从同一个PCD文件创建source和target（通过已知的旋转和平移）
2. 添加partial遮挡来测试partial registration能力
3. 提供ground truth用于精确评估

Usage:
    # 基本用法：旋转+平移
    python create_synthetic_test.py --input eagleg7/enhanced/1696641884.835595373.pcd --output synthetic_test/ --rotation 30 0 15 --translation 2.0 1.0 0.5
    
    # 添加partial遮挡
    python create_synthetic_test.py \
        --input eagleg7/enhanced/1706001766.376780611.pcd \
        --output synthetic_test/ \
        --rotation 45 0 0 \
        --translation 3.0 0 0 \
        --partial \
        --crop_src right \
        --crop_ref left \
        --crop_ratio 0.3

    # Partial测试（source右侧遮挡30%，reference左侧遮挡30%）
    python create_synthetic_test.py --input eagleg7/enhanced/1696641884.835595373.pcd     --output synthetic_partial/    --rotation 45 0 0     --translation 3.0 0 0  --partial  --crop_src right --crop_ref left --crop_ratio 0.3

    # 测试
    python infer_single_pair.py --h5 synthetic_test/synthetic_test.h5 --resume D:\AA_projects_in_nus\nus\deep_sparse_radar_odometry\code\checkpoints\partial-trained.pth --src 0 --ref 1    --num_iter 5 --auto_radius --neighbors 40 --save_vis

    # 对比ground truth
    cat synthetic_partial/ground_truth_transform.txt  # 真实值
    cat synthetic_partial/T_src0_ref1.txt            # 预测值
"""
import numpy as np
import open3d as o3d
import h5py
import os
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def estimate_normals(points: np.ndarray, k: int = 20):
    """估计法向量"""
    N = len(points)
    if N < k:
        # 点数不足，返回随机法向量
        normals = np.random.randn(N, 3)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        return normals
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    normals = np.zeros((N, 3))
    for i in range(N):
        neighbor_points = points[indices[i]]
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        
        if centered.shape[0] < 3:
            n = np.random.randn(3)
            n /= (np.linalg.norm(n) + 1e-12)
            normals[i] = n
            continue
        
        cov = np.cov(centered.T)
        if not np.all(np.isfinite(cov)):
            n = np.random.randn(3)
            n /= (np.linalg.norm(n) + 1e-12)
            normals[i] = n
            continue
        
        evals, evecs = np.linalg.eigh(cov)
        n = evecs[:, 0]  # 最小特征值对应的特征向量
        n /= (np.linalg.norm(n) + 1e-12)
        normals[i] = n
    
    return normals


def create_transform_matrix(rotation_deg: tuple, translation: tuple):
    """创建SE(3)变换矩阵
    
    Args:
        rotation_deg: (rx, ry, rz) 欧拉角（度）
        translation: (tx, ty, tz) 平移（米）
    
    Returns:
        T: (4, 4) 变换矩阵
    """
    # 创建旋转矩阵
    r = Rotation.from_euler('xyz', rotation_deg, degrees=True)
    R = r.as_matrix()
    
    # 创建平移向量
    t = np.array(translation)
    
    # 组合为4x4矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def apply_transform(points: np.ndarray, T: np.ndarray):
    """应用变换到点云"""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def crop_point_cloud(points: np.ndarray, direction: str, ratio: float):
    """裁剪点云以模拟partial观测
    
    Args:
        points: 点云 (N, 3)
        direction: 'left', 'right', 'front', 'back', 'top', 'bottom'
        ratio: 要移除的比例 (0-1)
    
    Returns:
        cropped_points: 裁剪后的点云
        mask: 保留点的mask
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    if direction == 'left':
        # 移除X轴负方向
        threshold = np.percentile(centered[:, 0], ratio * 100)
        mask = centered[:, 0] > threshold
    elif direction == 'right':
        # 移除X轴正方向
        threshold = np.percentile(centered[:, 0], 100 - ratio * 100)
        mask = centered[:, 0] < threshold
    elif direction == 'front':
        # 移除Y轴正方向
        threshold = np.percentile(centered[:, 1], 100 - ratio * 100)
        mask = centered[:, 1] < threshold
    elif direction == 'back':
        # 移除Y轴负方向
        threshold = np.percentile(centered[:, 1], ratio * 100)
        mask = centered[:, 1] > threshold
    elif direction == 'top':
        # 移除Z轴正方向
        threshold = np.percentile(centered[:, 2], 100 - ratio * 100)
        mask = centered[:, 2] < threshold
    elif direction == 'bottom':
        # 移除Z轴负方向
        threshold = np.percentile(centered[:, 2], ratio * 100)
        mask = centered[:, 2] > threshold
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return points[mask], mask


def create_synthetic_pair(
    pcd_file: str,
    output_dir: str,
    rotation_deg: tuple = (30, 0, 15),
    translation: tuple = (2.0, 1.0, 0.5),
    downsample_points: int = 1024,
    partial: bool = False,
    crop_src: str = 'right',
    crop_ref: str = 'left',
    crop_ratio: float = 0.3,
    k_normal: int = 30
):
    """创建合成测试对
    
    Args:
        pcd_file: 输入PCD文件
        output_dir: 输出目录
        rotation_deg: 旋转角度 (rx, ry, rz) 度
        translation: 平移 (tx, ty, tz) 米
        downsample_points: 下采样点数
        partial: 是否添加partial遮挡
        crop_src: source裁剪方向
        crop_ref: reference裁剪方向
        crop_ratio: 裁剪比例
        k_normal: 法向量估计邻居数
    """
    print("="*60)
    print("Creating Synthetic Test Data")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取PCD文件
    print(f"\nLoading PCD: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    points_original = np.asarray(pcd.points)
    print(f"Original points: {len(points_original)}")
    
    if len(points_original) == 0:
        raise ValueError("PCD file is empty!")
    
    # 2. 下采样到固定点数
    if len(points_original) > downsample_points:
        indices = np.random.choice(len(points_original), downsample_points, replace=False)
        points_base = points_original[indices]
    else:
        # 点数不足，重复采样
        repeat_times = downsample_points // len(points_original)
        remainder = downsample_points % len(points_original)
        points_base = np.tile(points_original, (repeat_times, 1))
        if remainder > 0:
            points_base = np.vstack([points_base, points_original[:remainder]])
    
    print(f"Downsampled to: {len(points_base)} points")
    
    # 3. 创建source（可选partial）
    if partial:
        points_src, mask_src = crop_point_cloud(points_base, crop_src, crop_ratio)
        print(f"Source after '{crop_src}' crop ({crop_ratio*100:.0f}%): {len(points_src)} points")
    else:
        points_src = points_base.copy()
        mask_src = np.ones(len(points_base), dtype=bool)
    
    # 4. 创建ground truth变换
    T_gt = create_transform_matrix(rotation_deg, translation)
    print(f"\nGround Truth Transform:")
    print(f"  Rotation (xyz): {rotation_deg} degrees")
    print(f"  Translation: {translation} meters")
    
    # 5. 应用变换得到reference
    points_ref_full = apply_transform(points_base, T_gt)
    
    # 6. 对reference应用partial（如果需要）
    if partial:
        points_ref, mask_ref = crop_point_cloud(points_ref_full, crop_ref, crop_ratio)
        print(f"Reference after '{crop_ref}' crop ({crop_ratio*100:.0f}%): {len(points_ref)} points")
    else:
        points_ref = points_ref_full
        mask_ref = np.ones(len(points_ref_full), dtype=bool)
    
    # 7. 估计法向量
    print(f"\nEstimating normals (k={k_normal})...")
    normals_src = estimate_normals(points_src, k=k_normal)
    normals_ref = estimate_normals(points_ref, k=k_normal)
    
    # 8. 保存为HDF5格式（compatible with RPM-Net）
    h5_file = os.path.join(output_dir, 'synthetic_test.h5')
    with h5py.File(h5_file, 'w') as f:
        # 保存为(2, N, 3)格式：2帧，N点，3维
        # Frame 0: source, Frame 1: reference
        data_array = np.stack([
            np.pad(points_src, ((0, downsample_points - len(points_src)), (0, 0)), mode='edge'),
            np.pad(points_ref, ((0, downsample_points - len(points_ref)), (0, 0)), mode='edge')
        ])
        normals_array = np.stack([
            np.pad(normals_src, ((0, downsample_points - len(normals_src)), (0, 0)), mode='edge'),
            np.pad(normals_ref, ((0, downsample_points - len(normals_ref)), (0, 0)), mode='edge')
        ])
        
        f.create_dataset('data', data=data_array)
        f.create_dataset('normal', data=normals_array)
        f.create_dataset('label', data=np.array([0, 1]))
        f.attrs['variable_points'] = False
        
        # 保存真实点数
        f.attrs['actual_points_src'] = len(points_src)
        f.attrs['actual_points_ref'] = len(points_ref)
    
    print(f"Saved HDF5: {h5_file}")
    
    # 9. 保存ground truth变换
    gt_file = os.path.join(output_dir, 'ground_truth_transform.txt')
    np.savetxt(gt_file, T_gt[:3, :])  # 保存3x4矩阵
    print(f"Saved ground truth: {gt_file}")
    
    # 10. 保存可视化PLY
    # Source (red)
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(points_src)
    pcd_src.normals = o3d.utility.Vector3dVector(normals_src)
    pcd_src.paint_uniform_color([1, 0, 0])
    o3d.io.write_point_cloud(os.path.join(output_dir, 'source.ply'), pcd_src)
    
    # Reference (blue)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(points_ref)
    pcd_ref.normals = o3d.utility.Vector3dVector(normals_ref)
    pcd_ref.paint_uniform_color([0, 0, 1])
    o3d.io.write_point_cloud(os.path.join(output_dir, 'reference.ply'), pcd_ref)
    
    # Combined
    o3d.io.write_point_cloud(os.path.join(output_dir, 'pair_ground_truth.ply'), pcd_src + pcd_ref)
    print(f"Saved PLY files: source.ply, reference.ply, pair_ground_truth.ply")
    
    # 11. 保存元数据
    meta_file = os.path.join(output_dir, 'test_metadata.txt')
    with open(meta_file, 'w') as f:
        f.write("Synthetic Test Metadata\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input PCD: {pcd_file}\n")
        f.write(f"Original points: {len(points_original)}\n")
        f.write(f"Downsampled points: {downsample_points}\n\n")
        
        f.write(f"Ground Truth Transform:\n")
        f.write(f"  Rotation (xyz): {rotation_deg} degrees\n")
        f.write(f"  Translation: {translation} meters\n\n")
        
        if partial:
            f.write(f"Partial Settings:\n")
            f.write(f"  Source crop: {crop_src} ({crop_ratio*100:.0f}%) -> {len(points_src)} points\n")
            f.write(f"  Reference crop: {crop_ref} ({crop_ratio*100:.0f}%) -> {len(points_ref)} points\n")
            f.write(f"  Overlap estimate: ~{(1-crop_ratio*2)*100:.0f}%\n\n")
        else:
            f.write(f"No partial crop (full point clouds)\n\n")
        
        f.write(f"Normal estimation: k={k_normal}\n")
    
    print(f"Saved metadata: {meta_file}")
    
    # 12. 保存test_files.txt和shape_names.txt（用于compatibility）
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        f.write("synthetic_test/synthetic_test.h5\n")
    
    with open(os.path.join(output_dir, 'shape_names.txt'), 'w') as f:
        f.write("synthetic_test\n")
    
    print("\n" + "="*60)
    print("Synthetic test data created successfully!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Test with infer_single_pair.py:")
    print(f"   python infer_single_pair.py \\")
    print(f"       --h5 {output_dir}/synthetic_test.h5 \\")
    print(f"       --resume /path/to/checkpoint.pth \\")
    print(f"       --src 0 --ref 1 \\")
    print(f"       --auto_radius --neighbors 30 --save_vis")
    print(f"\n2. Compare result with ground truth:")
    print(f"   cat {output_dir}/ground_truth_transform.txt")
    print(f"   cat {output_dir}/T_src0_ref1.txt  # After running inference")
    print(f"\n3. Visualize:")
    print(f"   Open {output_dir}/pair_ground_truth.ply to see ground truth")
    print(f"   Open {output_dir}/pair_after.ply to see RPM-Net result")


def main():
    parser = argparse.ArgumentParser(description='Create Synthetic Test Data for RPM-Net')
    
    # 输入输出
    parser.add_argument('--input', type=str, required=True,
                       help='Input PCD file path')
    parser.add_argument('--output', type=str, default='synthetic_test/',
                       help='Output directory')
    
    # 变换参数
    parser.add_argument('--rotation', nargs=3, type=float, default=[30, 0, 15],
                       help='Rotation in degrees (rx, ry, rz)')
    parser.add_argument('--translation', nargs=3, type=float, default=[2.0, 1.0, 0.5],
                       help='Translation in meters (tx, ty, tz)')
    
    # 下采样
    parser.add_argument('--downsample', type=int, default=1024,
                       help='Number of points to downsample to')
    
    # Partial设置
    parser.add_argument('--partial', action='store_true',
                       help='Enable partial point cloud (add occlusion)')
    parser.add_argument('--crop_src', type=str, default='right',
                       choices=['left', 'right', 'front', 'back', 'top', 'bottom'],
                       help='Crop direction for source')
    parser.add_argument('--crop_ref', type=str, default='left',
                       choices=['left', 'right', 'front', 'back', 'top', 'bottom'],
                       help='Crop direction for reference')
    parser.add_argument('--crop_ratio', type=float, default=0.3,
                       help='Ratio of points to remove (0-1)')
    
    # 其他
    parser.add_argument('--k_normal', type=int, default=30,
                       help='K for normal estimation')
    
    args = parser.parse_args()
    
    create_synthetic_pair(
        pcd_file=args.input,
        output_dir=args.output,
        rotation_deg=tuple(args.rotation),
        translation=tuple(args.translation),
        downsample_points=args.downsample,
        partial=args.partial,
        crop_src=args.crop_src,
        crop_ref=args.crop_ref,
        crop_ratio=args.crop_ratio,
        k_normal=args.k_normal
    )


if __name__ == '__main__':
    main()

