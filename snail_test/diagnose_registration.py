"""Diagnose registration issues and provide recommendations

Usage:
    python diagnose_registration.py --results_dir radar_single_frames_original/
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


class RegistrationDiagnostic:
    """配准诊断工具"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # 加载数据
        self.pred_transforms = np.load(self.results_dir / 'pred_transforms.npy')
        data_dict = np.load(self.results_dir / 'data_dict.npy', allow_pickle=True).item()
        self.points_src = data_dict['points_src']
        self.points_ref = data_dict['points_ref']
        
        print(f"Loaded data: {self.points_src.shape[0]} source points, {self.points_ref.shape[0]} reference points")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("\n" + "="*60)
        print("RADAR REGISTRATION DIAGNOSIS")
        print("="*60)
        
        # 1. 基本数据质量检查
        self._check_data_quality()
        
        # 2. 重叠度分析
        self._analyze_overlap()
        
        # 3. 变换分析
        self._analyze_transforms()
        
        # 4. 收敛性分析
        self._analyze_convergence()
        
        # 5. 参数建议
        self._provide_recommendations()
        
        # 6. 可视化诊断
        self._create_diagnostic_plots()
    
    def _check_data_quality(self):
        """检查数据质量"""
        print("\n📊 DATA QUALITY CHECK")
        print("-" * 30)
        
        # 点云统计
        src_center = np.mean(self.points_src, axis=0)
        ref_center = np.mean(self.points_ref, axis=0)
        src_scale = np.std(self.points_src, axis=0)
        ref_scale = np.std(self.points_ref, axis=0)
        
        print(f"Source center: ({src_center[0]:.2f}, {src_center[1]:.2f}, {src_center[2]:.2f})")
        print(f"Reference center: ({ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f})")
        print(f"Source scale: ({src_scale[0]:.2f}, {src_scale[1]:.2f}, {src_scale[2]:.2f})")
        print(f"Reference scale: ({ref_scale[0]:.2f}, {ref_scale[1]:.2f}, {ref_scale[2]:.2f})")
        
        # 距离分析
        center_distance = np.linalg.norm(src_center - ref_center)
        print(f"Center distance: {center_distance:.2f}m")
        
        # 点密度
        src_density = len(self.points_src) / (np.prod(src_scale) * 8)  # 近似体积
        ref_density = len(self.points_ref) / (np.prod(ref_scale) * 8)
        print(f"Point density: Source={src_density:.2e}, Reference={ref_density:.2e}")
        
        # 质量评估
        if center_distance > 50:
            print("⚠️  WARNING: Very large center distance - may need better initialization")
        if src_density < 1e-6 or ref_density < 1e-6:
            print("⚠️  WARNING: Very low point density - consider accumulating more frames")
    
    def _analyze_overlap(self):
        """分析重叠度"""
        print("\n🔍 OVERLAP ANALYSIS")
        print("-" * 30)
        
        # 计算初始重叠度
        initial_overlap = self._compute_overlap_ratio(self.points_src, self.points_ref)
        print(f"Initial overlap: {initial_overlap:.3f}")
        
        # 计算最终重叠度
        final_transform = self.pred_transforms[0, -1]
        aligned_source = self.apply_transform(self.points_src, final_transform)
        final_overlap = self._compute_overlap_ratio(aligned_source, self.points_ref)
        print(f"Final overlap: {final_overlap:.3f}")
        
        # 重叠度变化
        overlap_improvement = final_overlap - initial_overlap
        print(f"Overlap improvement: {overlap_improvement:.3f}")
        
        if initial_overlap < 0.1:
            print("❌ CRITICAL: Very low initial overlap - registration will likely fail")
        elif initial_overlap < 0.3:
            print("⚠️  WARNING: Low initial overlap - consider closer frame pairs")
        elif overlap_improvement < 0.05:
            print("⚠️  WARNING: Small overlap improvement - may need better parameters")
        else:
            print("✅ Good overlap improvement")
    
    def _analyze_transforms(self):
        """分析变换"""
        print("\n🔄 TRANSFORM ANALYSIS")
        print("-" * 30)
        
        transforms = self.pred_transforms[0]  # (n_iter, 3, 4)
        
        rotations = []
        translations = []
        
        for i, transform in enumerate(transforms):
            rot_deg = self.compute_rotation_error(transform)
            trans_m = self.compute_translation_magnitude(transform)
            rotations.append(rot_deg)
            translations.append(trans_m)
            
            print(f"Iteration {i+1}: Rot={rot_deg:.3f}°, Trans={trans_m:.3f}m")
        
        # 分析变化趋势
        rot_change = rotations[-1] - rotations[0]
        trans_change = translations[-1] - translations[0]
        
        print(f"\nTotal change: Rot={rot_change:.3f}°, Trans={trans_change:.3f}m")
        
        if abs(rot_change) < 0.1 and abs(trans_change) < 0.01:
            print("⚠️  WARNING: Very small changes - may indicate:")
            print("   - Already well aligned")
            print("   - Parameters too conservative")
            print("   - Insufficient overlap")
        elif abs(rot_change) > 30 or abs(trans_change) > 10:
            print("⚠️  WARNING: Very large changes - may indicate:")
            print("   - Poor initialization")
            print("   - Wrong correspondences")
            print("   - Parameters too aggressive")
    
    def _analyze_convergence(self):
        """分析收敛性"""
        print("\n📈 CONVERGENCE ANALYSIS")
        print("-" * 30)
        
        transforms = self.pred_transforms[0]
        
        rotations = []
        translations = []
        
        for transform in transforms:
            rot_deg = self.compute_rotation_error(transform)
            trans_m = self.compute_translation_magnitude(transform)
            rotations.append(rot_deg)
            translations.append(trans_m)
        
        # 计算变化率
        rot_changes = np.diff(rotations)
        trans_changes = np.diff(translations)
        
        print("Iteration-to-iteration changes:")
        for i, (rot_change, trans_change) in enumerate(zip(rot_changes, trans_changes)):
            print(f"  {i+1}→{i+2}: Rot={rot_change:+.3f}°, Trans={trans_change:+.3f}m")
        
        # 收敛性判断
        final_rot_change = abs(rot_changes[-1]) if len(rot_changes) > 0 else 0
        final_trans_change = abs(trans_changes[-1]) if len(trans_changes) > 0 else 0
        
        if final_rot_change < 0.01 and final_trans_change < 0.001:
            print("✅ Good convergence - changes are small in final iterations")
        elif final_rot_change > 1.0 or final_trans_change > 0.1:
            print("⚠️  WARNING: Poor convergence - still changing significantly")
            print("   Consider increasing --num_iter")
        else:
            print("🔄 Moderate convergence - may benefit from more iterations")
    
    def _provide_recommendations(self):
        """提供参数建议"""
        print("\n💡 PARAMETER RECOMMENDATIONS")
        print("-" * 30)
        
        # 基于数据特征的建议
        src_scale = np.std(self.points_src, axis=0)
        avg_scale = np.mean(src_scale)
        
        # 重叠度
        initial_overlap = self._compute_overlap_ratio(self.points_src, self.points_ref)
        
        # 变换幅度
        transforms = self.pred_transforms[0]
        max_rot = max(self.compute_rotation_error(t) for t in transforms)
        max_trans = max(self.compute_translation_magnitude(t) for t in transforms)
        
        print("Based on your data analysis:")
        
        # Neighbors建议
        if avg_scale > 20:  # 大尺度场景
            print(f"🔧 --neighbors: Try 40-60 (large scale: {avg_scale:.1f}m)")
        elif avg_scale > 10:
            print(f"🔧 --neighbors: Try 30-50 (medium scale: {avg_scale:.1f}m)")
        else:
            print(f"🔧 --neighbors: Try 20-40 (small scale: {avg_scale:.1f}m)")
        
        # Radius建议
        if avg_scale > 20:
            print(f"🔧 --radius: Try 5-10m or --auto_radius (large scale)")
        elif avg_scale > 10:
            print(f"🔧 --radius: Try 2-5m or --auto_radius (medium scale)")
        else:
            print(f"🔧 --radius: Try 1-3m or --auto_radius (small scale)")
        
        # Iterations建议
        if max_rot > 10 or max_trans > 2:
            print(f"🔧 --num_iter: Try 10-15 (large initial misalignment)")
        else:
            print(f"🔧 --num_iter: Try 5-10 (moderate misalignment)")
        
        # 重叠度建议
        if initial_overlap < 0.2:
            print("🔧 Overlap: Consider using closer frame pairs or accumulating more frames")
        elif initial_overlap < 0.4:
            print("🔧 Overlap: Current overlap is acceptable but could be better")
        else:
            print("🔧 Overlap: Good overlap - current parameters should work well")
        
        # 具体命令建议
        print(f"\n🚀 Suggested command:")
        if avg_scale > 20:
            neighbors = 50
            radius_cmd = "--auto_radius"
        elif avg_scale > 10:
            neighbors = 40
            radius_cmd = "--auto_radius"
        else:
            neighbors = 30
            radius_cmd = "--auto_radius"
        
        if max_rot > 10 or max_trans > 2:
            num_iter = 12
        else:
            num_iter = 8
        
        print(f"python infer_single_pair.py \\")
        print(f"  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \\")
        print(f"  --resume checkpoints/partial-trained.pth \\")
        print(f"  --src 0 --ref 3 \\")
        print(f"  --num_iter {num_iter} \\")
        print(f"  --neighbors {neighbors} \\")
        print(f"  {radius_cmd} \\")
        print(f"  --save_vis")
    
    def _create_diagnostic_plots(self):
        """创建诊断图表"""
        print("\n📊 Creating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Registration Diagnostic Plots', fontsize=16)
        
        # 1. 变换曲线
        ax = axes[0, 0]
        transforms = self.pred_transforms[0]
        rotations = [self.compute_rotation_error(t) for t in transforms]
        translations = [self.compute_translation_magnitude(t) for t in transforms]
        
        iterations = range(1, len(transforms) + 1)
        ax.plot(iterations, rotations, 'ro-', label='Rotation (deg)')
        ax2 = ax.twinx()
        ax2.plot(iterations, translations, 'bo-', label='Translation (m)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rotation Error (degrees)', color='r')
        ax2.set_ylabel('Translation Error (meters)', color='b')
        ax.set_title('Transform Evolution')
        ax.grid(True, alpha=0.3)
        
        # 2. 点云分布
        ax = axes[0, 1]
        ax.scatter(self.points_src[:, 0], self.points_src[:, 1], 
                  c='red', s=5, alpha=0.6, label='Source')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=5, alpha=0.6, label='Reference')
        ax.set_title('Point Cloud Distribution')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 3. 对齐结果
        ax = axes[0, 2]
        final_transform = transforms[-1]
        aligned_source = self.apply_transform(self.points_src, final_transform)
        ax.scatter(aligned_source[:, 0], aligned_source[:, 1], 
                  c='green', s=5, alpha=0.6, label='Aligned Source')
        ax.scatter(self.points_ref[:, 0], self.points_ref[:, 1], 
                  c='blue', s=5, alpha=0.6, label='Reference')
        ax.set_title('Final Alignment')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 4. 重叠度变化
        ax = axes[1, 0]
        overlaps = []
        for transform in transforms:
            aligned_source = self.apply_transform(self.points_src, transform)
            overlap = self._compute_overlap_ratio(aligned_source, self.points_ref)
            overlaps.append(overlap)
        
        ax.plot(iterations, overlaps, 'go-', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overlap Ratio')
        ax.set_title('Overlap Improvement')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 5. 点密度热力图
        ax = axes[1, 1]
        all_points = np.vstack([self.points_src, self.points_ref])
        hist, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=30)
        im = ax.imshow(hist.T, extent=[all_points[:, 0].min(), all_points[:, 0].max(),
                                      all_points[:, 1].min(), all_points[:, 1].max()], 
                      origin='lower', cmap='hot', alpha=0.7)
        ax.set_title('Point Density')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax, label='Density')
        
        # 6. 误差分布
        ax = axes[1, 2]
        final_aligned = self.apply_transform(self.points_src, final_transform)
        distances = self._compute_point_distances(final_aligned, self.points_ref)
        ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Distance to Nearest Reference Point (m)')
        ax.set_ylabel('Count')
        ax.set_title('Final Alignment Error Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = self.results_dir / 'diagnostic_plots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plots saved to: {save_path}")
        
        plt.show()
    
    def _compute_overlap_ratio(self, points1, points2, threshold=2.0):
        """计算重叠比例"""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points2)
        distances, _ = nbrs.kneighbors(points1)
        close_points = np.sum(distances.flatten() < threshold)
        return close_points / len(points1)
    
    def _compute_point_distances(self, points1, points2):
        """计算点到最近点的距离"""
        if len(points1) == 0 or len(points2) == 0:
            return np.array([])
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points2)
        distances, _ = nbrs.kneighbors(points1)
        return distances.flatten()
    
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
    parser = argparse.ArgumentParser(description='Diagnose Radar Registration Issues')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory containing pred_transforms.npy and data_dict.npy')
    
    args = parser.parse_args()
    
    diagnostic = RegistrationDiagnostic(args.results_dir)
    diagnostic.run_full_diagnosis()


if __name__ == '__main__':
    main()
