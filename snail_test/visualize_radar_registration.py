"""Visualize RPM-Net registration results for radar data

Usage:
    python visualize_radar_registration.py \
        --results_dir radar_single_frames_original/ \
        --mode progress

Modes:
    - progress: Show registration progress across iterations
    - 3d: 3D visualization with Open3D
    - matching: Visualize matching matrix

python visualize_radar_registration.py  --results_dir synthetic_test/  --mode progress --save progress.png
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from pathlib import Path


class RadarRegistrationVisualizer:
    """可视化雷达数据的RPM-Net配准结果"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 包含结果文件的目录 (pred_transforms.npy, perm_matrices.pickle, data_dict.npy)
        """
        self.results_dir = Path(results_dir)
        
        # 加载结果
        self.pred_transforms = np.load(self.results_dir / 'pred_transforms.npy')  # (1, n_iter, 3, 4)
        
        # 加载permutation matrices (如果存在)
        perm_path = self.results_dir / 'perm_matrices.pickle'
        if perm_path.exists():
            with open(perm_path, 'rb') as f:
                self.perm_matrices = pickle.load(f)  # List of [list of sparse matrices]
        else:
            self.perm_matrices = None
            print("Warning: perm_matrices.pickle not found")
        
        # 加载数据
        data_dict_path = self.results_dir / 'data_dict.npy'
        if data_dict_path.exists():
            data_dict = np.load(data_dict_path, allow_pickle=True).item()
            self.points_src = data_dict['points_src']
            self.points_ref = data_dict['points_ref']
            self.normals_src = data_dict.get('normals_src', None)
            self.normals_ref = data_dict.get('normals_ref', None)
        else:
            print("Error: data_dict.npy not found!")
            raise FileNotFoundError(f"Required file not found: {data_dict_path}")
        
        print(f"Loaded transforms: {self.pred_transforms.shape}")
        print(f"Source points: {self.points_src.shape}")
        print(f"Reference points: {self.points_ref.shape}")
    
    def visualize_registration_progress(self, save_path: str = None):
        """可视化配准迭代过程"""
        
        transforms = self.pred_transforms[0]  # (n_iter, 3, 4)
        n_iter = transforms.shape[0]
        
        # 创建子图: 初始 + 每次迭代
        n_cols = min(3, n_iter + 1)
        n_rows = (n_iter + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        fig.suptitle('RPM-Net Registration Progress (Radar Data)', fontsize=16)
        
        # 初始状态
        ax = axes[0]
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=5, alpha=0.6, label='Source')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=5, alpha=0.6, label='Reference')
        ax.set_title('Initial (Before Registration)')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 每次迭代
        for iter_idx in range(n_iter):
            ax = axes[iter_idx + 1]
            
            # 应用变换
            transform = transforms[iter_idx]
            aligned_source = self.apply_transform(self.points_src, transform)
            
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], 
                      c='green', s=5, alpha=0.6, label='Aligned Source')
            ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                      c='blue', s=5, alpha=0.6, label='Reference')
            
            # 计算误差
            rotation_deg = self.compute_rotation_error(transform)
            translation_m = self.compute_translation_magnitude(transform)
            
            ax.set_title(f'Iteration {iter_idx + 1}\nRot: {rotation_deg:.2f}°, Trans: {translation_m:.3f}m')
            ax.legend()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        
        # 隐藏多余的子图
        for idx in range(n_iter + 1, len(axes)):
            axes[idx].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def visualize_3d_registration(self, iteration: int = -1):
        """3D可视化配准结果
        
        Args:
            iteration: 要可视化的迭代编号 (-1表示最后一次)
        """
        if iteration < 0:
            iteration = self.pred_transforms.shape[1] - 1
        
        transform = self.pred_transforms[0, iteration]
        
        # 应用变换
        aligned_source = self.apply_transform(self.points_src, transform)
        
        # 创建Open3D点云
        pcd_source_original = o3d.geometry.PointCloud()
        pcd_source_original.points = o3d.utility.Vector3dVector(self.points_src)
        pcd_source_original.paint_uniform_color([1, 0, 0])  # 红色 - 原始source
        
        pcd_source_aligned = o3d.geometry.PointCloud()
        pcd_source_aligned.points = o3d.utility.Vector3dVector(aligned_source)
        pcd_source_aligned.paint_uniform_color([0, 1, 0])  # 绿色 - 对齐后的source
        
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(self.points_ref)
        pcd_target.paint_uniform_color([0, 0, 1])  # 蓝色 - reference
        
        # 添加法向量（如果有）
        if self.normals_src is not None:
            pcd_source_original.normals = o3d.utility.Vector3dVector(self.normals_src)
            # 变换法向量
            R = transform[:, :3]
            normals_aligned = self.normals_src @ R.T
            pcd_source_aligned.normals = o3d.utility.Vector3dVector(normals_aligned)
        if self.normals_ref is not None:
            pcd_target.normals = o3d.utility.Vector3dVector(self.normals_ref)
        
        # 可视化
        print("\n3D Visualization:")
        print("  Red: Original Source")
        print("  Green: Aligned Source")
        print("  Blue: Reference")
        print("\nPress 'N' to show/hide normals")
        
        o3d.visualization.draw_geometries(
            [pcd_source_aligned, pcd_target],
            window_name=f"3D Registration - Iteration {iteration+1}",
            point_show_normal=False
        )
    
    def visualize_matching_matrix(self, iteration: int = -1, save_path: str = None):
        """可视化匹配矩阵
        
        Args:
            iteration: 要可视化的迭代编号 (-1表示最后一次)
        """
        if self.perm_matrices is None:
            print("Error: Permutation matrices not available")
            return
        
        if iteration < 0:
            iteration = len(self.perm_matrices[0]) - 1
        
        # 获取稀疏矩阵并转为密集矩阵
        perm_matrix = self.perm_matrices[0][iteration].toarray()
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 完整矩阵
        im1 = ax1.imshow(perm_matrix, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Matching Weight')
        ax1.set_title(f'Full Matching Matrix - Iteration {iteration+1}')
        ax1.set_xlabel('Reference Points')
        ax1.set_ylabel('Source Points')
        
        # 阈值化后的矩阵（突出显示强匹配）
        threshold = np.percentile(perm_matrix, 95)
        perm_matrix_thresh = np.where(perm_matrix > threshold, perm_matrix, 0)
        im2 = ax2.imshow(perm_matrix_thresh, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Matching Weight')
        ax2.set_title(f'Strong Matches (>95th percentile) - Iteration {iteration+1}')
        ax2.set_xlabel('Reference Points')
        ax2.set_ylabel('Source Points')
        
        # 统计信息
        n_strong_matches = np.sum(perm_matrix_thresh > 0)
        avg_weight = np.mean(perm_matrix[perm_matrix > threshold]) if n_strong_matches > 0 else 0
        
        fig.text(0.5, 0.02, 
                f'Strong matches: {n_strong_matches} | Avg weight: {avg_weight:.4f} | Threshold: {threshold:.4f}',
                ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def create_iteration_comparison(self, save_path: str = None):
        """创建所有迭代的对比图（top view）"""
        transforms = self.pred_transforms[0]  # (n_iter, 3, 4)
        n_iter = transforms.shape[0]
        
        fig, axes = plt.subplots(1, n_iter + 1, figsize=(5*(n_iter+1), 5))
        fig.suptitle('Registration Progress - Top View (XY)', fontsize=16)
        
        # 初始状态
        ax = axes[0]
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=3, alpha=0.6, label='Source')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=3, alpha=0.6, label='Reference')
        ax.set_title('Initial')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 每次迭代
        for iter_idx in range(n_iter):
            ax = axes[iter_idx + 1]
            transform = transforms[iter_idx]
            aligned_source = self.apply_transform(self.points_src, transform)
            
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], 
                      c='green', s=3, alpha=0.6, label='Aligned')
            ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                      c='blue', s=3, alpha=0.6, label='Reference')
            
            rotation_deg = self.compute_rotation_error(transform)
            translation_m = self.compute_translation_magnitude(transform)
            
            ax.set_title(f'Iter {iter_idx + 1}\n{rotation_deg:.2f}°, {translation_m:.3f}m')
            ax.legend()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """应用SE(3)变换"""
        R = transform[:3, :3]
        t = transform[:3, 3]
        return points @ R.T + t
    
    def compute_rotation_error(self, transform: np.ndarray) -> float:
        """计算旋转角度（度）"""
        R = transform[:3, :3]
        trace = np.trace(R)
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        return float(angle_rad * 180 / np.pi)
    
    def compute_translation_magnitude(self, transform: np.ndarray) -> float:
        """计算平移距离（米）"""
        t = transform[:3, 3]
        return float(np.linalg.norm(t))


def main():
    parser = argparse.ArgumentParser(description='Visualize Radar Registration Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results (pred_transforms.npy, etc.)')
    parser.add_argument('--mode', type=str, 
                       choices=['progress', '3d', 'matching', 'comparison'], 
                       default='progress',
                       help='Visualization mode')
    parser.add_argument('--iteration', type=int, default=-1,
                       help='Iteration to visualize (-1 for last, only for 3d/matching modes)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the figure')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = RadarRegistrationVisualizer(args.results_dir)
    
    # 根据模式可视化
    if args.mode == 'progress':
        visualizer.visualize_registration_progress(save_path=args.save)
    elif args.mode == '3d':
        visualizer.visualize_3d_registration(iteration=args.iteration)
    elif args.mode == 'matching':
        visualizer.visualize_matching_matrix(iteration=args.iteration, save_path=args.save)
    elif args.mode == 'comparison':
        visualizer.create_iteration_comparison(save_path=args.save)


if __name__ == '__main__':
    main()

