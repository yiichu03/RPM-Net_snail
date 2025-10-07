'''
python visualize_rpmnet.py --mode progress --sample_idx 0 --eval_results ../eval_results --dataset_path ../datasets/modelnet40_ply_hdf5_2048
python visualize_rpmnet.py --mode 3d --sample_idx 0
'''

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from typing import List, Tuple
import torch
from scipy.spatial.transform import Rotation
import matplotlib.patches as patches
import sys
sys.path.append('src')  # 添加src目录到Python路径

class RPMNetVisualizer:
    """RPMNet结果可视化工具"""
    
    def __init__(self, eval_results_path: str, dataset_path: str):
        """
        Args:
            eval_results_path: eval_results文件夹路径
            dataset_path: ModelNet40数据集路径
        """
        self.eval_results_path = eval_results_path
        self.dataset_path = dataset_path
        
        # 加载评估结果
        self.pred_transforms = np.load(os.path.join(eval_results_path, 'pred_transforms.npy'))
        with open(os.path.join(eval_results_path, 'perm_matrices.pickle'), 'rb') as f:
            self.perm_matrices = pickle.load(f)
        
        print(f"Loaded transforms shape: {self.pred_transforms.shape}")
        print(f"Loaded {len(self.perm_matrices)} permutation matrices")
        
    def load_point_cloud_data(self, sample_idx: int):
        """加载指定样本的点云数据"""
        # 这里需要重新运行数据加载器来获取原始点云
        # 由于数据加载器包含随机变换，我们需要重新实现确定性的数据加载
        from data_loader.datasets import get_test_datasets
        from arguments import rpmnet_eval_arguments
        
        # 创建参数
        parser = rpmnet_eval_arguments()
        args = parser.parse_args(['--noise_type', 'crop', '--dataset_path', self.dataset_path])
        
        # 获取测试数据集
        test_dataset = get_test_datasets(args)
        
        # 获取指定样本
        sample = test_dataset[sample_idx]
        
        return {
            'points_src': sample['points_src'][:, :3],  # 只要xyz坐标
            'points_ref': sample['points_ref'][:, :3],
            'points_raw': sample['points_raw'][:, :3],
            'transform_gt': sample['transform_gt']
        }
    
    def visualize_registration_progress(self, sample_idx: int, save_path: str = None):
        """可视化配准过程 - 显示每次迭代的结果"""
        
        # 加载点云数据
        data = self.load_point_cloud_data(sample_idx)
        
        # 获取变换矩阵
        transforms = self.pred_transforms[sample_idx]  # (5, 3, 4)
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'RPMNet Registration Progress - Sample {sample_idx}', fontsize=16)
        
        # 颜色定义
        colors = {
            'source': 'red',
            'target': 'blue', 
            'aligned': 'green'
        }
        
        # 初始状态 (迭代0)
        ax = axes[0, 0]
        source_pts = data['points_src']
        target_pts = data['points_ref']
        
        ax.scatter(source_pts[:, 0], source_pts[:, 1], c=colors['source'], s=1, alpha=0.6, label='Source')
        ax.scatter(target_pts[:, 0], target_pts[:, 1], c=colors['target'], s=1, alpha=0.6, label='Target')
        ax.set_title('Initial (Iter 0)')
        ax.legend()
        ax.set_aspect('equal')
        
        # 每次迭代的结果
        # 修改为：
        for iter_idx in range(5):
            if iter_idx < 2:  # Iter 1, 2
                row, col = 0, iter_idx + 1
            else:  # Iter 3, 4, 5  
                row, col = 1, iter_idx - 2            
            if row >= 2:
                break
                
            ax = axes[row, col]
            
            # 应用变换
            transform = transforms[iter_idx]
            aligned_source = self.apply_transform(source_pts, transform)
            
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], c=colors['aligned'], s=1, alpha=0.6, label='Aligned Source')
            ax.scatter(target_pts[:, 0], target_pts[:, 1], c=colors['target'], s=1, alpha=0.6, label='Target')
            ax.set_title(f'Iter {iter_idx + 1}')
            ax.legend()
            ax.set_aspect('equal')
        
        # 移除多余的子图
        if len(transforms) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_3d_registration(self, sample_idx: int, iteration: int = 4):
        """3D可视化配准结果"""
        
        # 加载数据
        data = self.load_point_cloud_data(sample_idx)
        transform = self.pred_transforms[sample_idx, iteration]
        
        # 应用变换
        aligned_source = self.apply_transform(data['points_src'], transform)
        
        # 创建Open3D点云
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(aligned_source)
        pcd_source.paint_uniform_color([1, 0, 0])  # 红色 - 变换后的source
        
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(data['points_ref'])
        pcd_target.paint_uniform_color([0, 0, 1])  # 蓝色 - target
        
        # 可视化
        o3d.visualization.draw_geometries([pcd_source, pcd_target],
                                        window_name=f"3D Registration - Sample {sample_idx}, Iter {iteration+1}")
    
    def visualize_matching_matrix(self, sample_idx: int, iteration: int = 4):
        """可视化匹配矩阵"""
        
        if sample_idx >= len(self.perm_matrices):
            print(f"Sample index {sample_idx} out of range")
            return
            
        # 获取匹配矩阵
        perm_matrix = self.perm_matrices[sample_idx][iteration].toarray()
        
        # 可视化
        plt.figure(figsize=(10, 8))
        plt.imshow(perm_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Matching Weight')
        plt.title(f'Matching Matrix - Sample {sample_idx}, Iter {iteration+1}')
        plt.xlabel('Target Points')
        plt.ylabel('Source Points')
        plt.show()
    
    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """应用SE(3)变换到点云"""
        # transform shape: (3, 4)
        R = transform[:3, :3]  # 旋转矩阵
        t = transform[:3, 3]   # 平移向量
        
        return points @ R.T + t
    
    def create_comparison_video(self, sample_indices: List[int], save_path: str = None):
        """创建多个样本的对比视频/图像"""
        
        n_samples = len(sample_indices)
        fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 10))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i, sample_idx in enumerate(sample_indices):
            # 加载数据
            data = self.load_point_cloud_data(sample_idx)
            
            # 初始状态
            ax = axes[0, i]
            source_pts = data['points_src']
            target_pts = data['points_ref']
            
            ax.scatter(source_pts[:, 0], source_pts[:, 1], c='red', s=1, alpha=0.6, label='Source')
            ax.scatter(target_pts[:, 0], target_pts[:, 1], c='blue', s=1, alpha=0.6, label='Target')
            ax.set_title(f'Sample {sample_idx} - Before')
            ax.legend()
            ax.set_aspect('equal')
            
            # 最终结果
            ax = axes[1, i]
            final_transform = self.pred_transforms[sample_idx, -1]
            aligned_source = self.apply_transform(source_pts, final_transform)
            
            ax.scatter(aligned_source[:, 0], aligned_source[:, 1], c='green', s=1, alpha=0.6, label='Aligned')
            ax.scatter(target_pts[:, 0], target_pts[:, 1], c='blue', s=1, alpha=0.6, label='Target')
            ax.set_title(f'Sample {sample_idx} - After')
            ax.legend()
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='RPMNet Visualization Tool')
    parser.add_argument('--eval_results', type=str, default='../eval_results',
                       help='Path to evaluation results folder')
    parser.add_argument('--dataset_path', type=str, default='../datasets/modelnet40_ply_hdf5_2048',
                       help='Path to ModelNet40 dataset')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--mode', type=str, choices=['progress', '3d', 'matching', 'comparison'], 
                       default='progress', help='Visualization mode')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = RPMNetVisualizer(args.eval_results, args.dataset_path)
    
    if args.mode == 'progress':
        visualizer.visualize_registration_progress(args.sample_idx)
    elif args.mode == '3d':
        visualizer.visualize_3d_registration(args.sample_idx)
    elif args.mode == 'matching':
        visualizer.visualize_matching_matrix(args.sample_idx)
    elif args.mode == 'comparison':
        visualizer.create_comparison_video([args.sample_idx, args.sample_idx+1, args.sample_idx+2])

if __name__ == '__main__':
    main()