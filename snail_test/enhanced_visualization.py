"""Enhanced visualization for radar registration - better visibility of changes

Usage:
    python enhanced_visualization.py  --results_dir radar_single_frames_original/ --mode enhanced_progress
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from pathlib import Path


class EnhancedRadarVisualizer:
    """增强版雷达配准可视化 - 更好地显示变化"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # 加载数据
        self.pred_transforms = np.load(self.results_dir / 'pred_transforms.npy')
        data_dict = np.load(self.results_dir / 'data_dict.npy', allow_pickle=True).item()
        self.points_src = data_dict['points_src']
        self.points_ref = data_dict['points_ref']
        
        print(f"Loaded transforms: {self.pred_transforms.shape}")
        print(f"Source points: {self.points_src.shape}")
        print(f"Reference points: {self.points_ref.shape}")
    
    def enhanced_progress_visualization(self, save_path: str = None):
        """增强版进度可视化 - 多种视角和放大"""
        
        transforms = self.pred_transforms[0]  # (n_iter, 3, 4)
        n_iter = transforms.shape[0]
        
        # 创建4个子图：原始、放大、误差曲线、变换分解
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原始视图（所有迭代）
        ax1 = plt.subplot(2, 3, 1)
        self._plot_all_iterations(ax1, transforms, "All Iterations - Full View")
        
        # 2. 放大视图（聚焦中心区域）
        ax2 = plt.subplot(2, 3, 2)
        self._plot_zoomed_iterations(ax2, transforms, "Zoomed View - Center Region")
        
        # 3. 误差曲线
        ax3 = plt.subplot(2, 3, 3)
        self._plot_error_curves(ax3, transforms)
        
        # 4. 变换分解（旋转和平移）
        ax4 = plt.subplot(2, 3, 4)
        self._plot_transform_components(ax4, transforms)
        
        # 5. 点云密度分析
        ax5 = plt.subplot(2, 3, 5)
        self._plot_point_density(ax5)
        
        # 6. 重叠度分析
        ax6 = plt.subplot(2, 3, 6)
        self._plot_overlap_analysis(ax6, transforms)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced visualization saved to: {save_path}")
        
        plt.show()
    
    def _plot_all_iterations(self, ax, transforms, title):
        """绘制所有迭代的完整视图"""
        colors = plt.cm.viridis(np.linspace(0, 1, len(transforms)))
        
        # 初始状态
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=8, alpha=0.7, label='Source (Initial)', marker='o')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=8, alpha=0.7, label='Reference', marker='s')
        
        # 每次迭代
        for i, (transform, color) in enumerate(zip(transforms, colors)):
            aligned_source = self.apply_transform(self.points_src, transform)
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], 
                      c=color, s=4, alpha=0.5, 
                      label=f'Iter {i+1}' if i < 3 else None)
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    def _plot_zoomed_iterations(self, ax, transforms, title):
        """绘制放大的中心区域"""
        # 计算中心点
        all_points = np.vstack([self.points_src, self.points_ref])
        center = np.mean(all_points, axis=0)
        
        # 计算合适的缩放范围（覆盖80%的点）
        distances = np.linalg.norm(all_points - center, axis=1)
        radius = np.percentile(distances, 80)
        
        # 绘制放大的视图
        colors = plt.cm.viridis(np.linspace(0, 1, len(transforms)))
        
        # 初始状态
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=15, alpha=0.8, label='Source (Initial)', marker='o')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=15, alpha=0.8, label='Reference', marker='s')
        
        # 每次迭代（更明显的标记）
        for i, (transform, color) in enumerate(zip(transforms, colors)):
            aligned_source = self.apply_transform(self.points_src, transform)
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], 
                      c=color, s=8, alpha=0.7, 
                      label=f'Iter {i+1}' if i < 3 else None,
                      marker='^' if i % 2 == 0 else 'v')
        
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    def _plot_error_curves(self, ax, transforms):
        """绘制误差曲线"""
        rotations = []
        translations = []
        
        for transform in transforms:
            rot_deg = self.compute_rotation_error(transform)
            trans_m = self.compute_translation_magnitude(transform)
            rotations.append(rot_deg)
            translations.append(trans_m)
        
        iterations = range(1, len(transforms) + 1)
        
        ax.plot(iterations, rotations, 'ro-', linewidth=2, markersize=8, label='Rotation (deg)')
        ax.plot(iterations, translations, 'bo-', linewidth=2, markersize=8, label='Translation (m)')
        
        ax.set_title('Error Reduction Over Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 对数尺度更好地显示变化
    
    def _plot_transform_components(self, ax, transforms):
        """绘制变换分量"""
        rotations = []
        translations = []
        
        for transform in transforms:
            rot_deg = self.compute_rotation_error(transform)
            trans_m = self.compute_translation_magnitude(transform)
            rotations.append(rot_deg)
            translations.append(trans_m)
        
        iterations = range(1, len(transforms) + 1)
        
        # 双y轴
        ax2 = ax.twinx()
        
        line1 = ax.plot(iterations, rotations, 'r-o', linewidth=2, markersize=6, label='Rotation (deg)')
        line2 = ax2.plot(iterations, translations, 'b-s', linewidth=2, markersize=6, label='Translation (m)')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rotation Error (degrees)', color='r')
        ax2.set_ylabel('Translation Error (meters)', color='b')
        ax.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.set_title('Transform Components')
        ax.grid(True, alpha=0.3)
    
    def _plot_point_density(self, ax):
        """分析点云密度"""
        # 计算2D直方图
        all_points = np.vstack([self.points_src, self.points_ref])
        
        # 创建网格
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # 2D直方图
        hist, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=50)
        
        # 绘制热力图
        im = ax.imshow(hist.T, extent=[x_min, x_max, y_min, y_max], 
                      origin='lower', cmap='hot', alpha=0.7)
        
        # 叠加点云
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=3, alpha=0.6, label='Source')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=3, alpha=0.6, label='Reference')
        
        ax.set_title('Point Cloud Density Analysis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Point Density')
    
    def _plot_overlap_analysis(self, ax, transforms):
        """分析重叠度变化"""
        overlaps = []
        
        for i, transform in enumerate(transforms):
            aligned_source = self.apply_transform(self.points_src, transform)
            overlap = self._compute_overlap_ratio(aligned_source, self.points_ref)
            overlaps.append(overlap)
        
        iterations = range(1, len(transforms) + 1)
        
        ax.plot(iterations, overlaps, 'go-', linewidth=2, markersize=8)
        ax.set_title('Overlap Ratio Over Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overlap Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _compute_overlap_ratio(self, points1, points2, threshold=2.0):
        """计算两个点云的重叠比例"""
        from sklearn.neighbors import NearestNeighbors
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # 对每个点1，找最近的点2
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points2)
        distances, _ = nbrs.kneighbors(points1)
        
        # 计算在阈值内的点比例
        close_points = np.sum(distances.flatten() < threshold)
        return close_points / len(points1)
    
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
    parser = argparse.ArgumentParser(description='Enhanced Radar Registration Visualization')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--mode', type=str, default='enhanced_progress',
                       help='Visualization mode')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the figure')
    
    args = parser.parse_args()
    
    visualizer = EnhancedRadarVisualizer(args.results_dir)
    
    if args.mode == 'enhanced_progress':
        visualizer.enhanced_progress_visualization(save_path=args.save)


if __name__ == '__main__':
    main()
