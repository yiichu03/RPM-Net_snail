import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_samples(eval_results_path='../eval_results'):
    """分析评估结果，找出有代表性的样本"""
    
    # 读取Excel文件中的详细指标
    excel_file = os.path.join(eval_results_path, 'metrics.xlsx')
    
    # 读取最后一次迭代的结果
    df = pd.read_excel(excel_file, sheet_name='Iter_5')
    
    print(f"总共分析了 {len(df)} 个样本")
    print(f"指标列: {list(df.columns)}")
    
    # 分析旋转误差
    rotation_errors = df['err_r_deg'].values
    translation_errors = df['err_t'].values
    chamfer_distances = df['chamfer_dist'].values
    
    # 找出表现最好和最差的样本
    best_rotation_idx = np.argmin(rotation_errors)
    worst_rotation_idx = np.argmax(rotation_errors)
    
    best_translation_idx = np.argmin(translation_errors)
    worst_translation_idx = np.argmax(translation_errors)
    
    best_chamfer_idx = np.argmin(chamfer_distances)
    worst_chamfer_idx = np.argmax(chamfer_distances)
    
    print("\n=== 推荐样本 ===")
    print(f"旋转误差最小的样本: {best_rotation_idx} (误差: {rotation_errors[best_rotation_idx]:.4f}°)")
    print(f"旋转误差最大的样本: {worst_rotation_idx} (误差: {rotation_errors[worst_rotation_idx]:.4f}°)")
    print(f"平移误差最小的样本: {best_translation_idx} (误差: {translation_errors[best_translation_idx]:.4f})")
    print(f"平移误差最大的样本: {worst_translation_idx} (误差: {translation_errors[worst_translation_idx]:.4f})")
    print(f"Chamfer距离最小的样本: {best_chamfer_idx} (距离: {chamfer_distances[best_chamfer_idx]:.6f})")
    print(f"Chamfer距离最大的样本: {worst_chamfer_idx} (距离: {chamfer_distances[worst_chamfer_idx]:.6f})")
    
    # 找出中等表现的样本
    median_rotation_idx = np.argsort(rotation_errors)[len(rotation_errors)//2]
    median_translation_idx = np.argsort(translation_errors)[len(translation_errors)//2]
    
    print(f"中等旋转误差的样本: {median_rotation_idx} (误差: {rotation_errors[median_rotation_idx]:.4f}°)")
    print(f"中等平移误差的样本: {median_translation_idx} (误差: {translation_errors[median_translation_idx]:.4f})")
    
    # 创建误差分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(rotation_errors, bins=50, alpha=0.7, color='blue')
    axes[0].axvline(rotation_errors[best_rotation_idx], color='green', linestyle='--', label='Best')
    axes[0].axvline(rotation_errors[worst_rotation_idx], color='red', linestyle='--', label='Worst')
    axes[0].set_xlabel('Rotation Error (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Rotation Error Distribution')
    axes[0].legend()
    
    axes[1].hist(translation_errors, bins=50, alpha=0.7, color='orange')
    axes[1].axvline(translation_errors[best_translation_idx], color='green', linestyle='--', label='Best')
    axes[1].axvline(translation_errors[worst_translation_idx], color='red', linestyle='--', label='Worst')
    axes[1].set_xlabel('Translation Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Translation Error Distribution')
    axes[1].legend()
    
    axes[2].hist(chamfer_distances, bins=50, alpha=0.7, color='purple')
    axes[2].axvline(chamfer_distances[best_chamfer_idx], color='green', linestyle='--', label='Best')
    axes[2].axvline(chamfer_distances[worst_chamfer_idx], color='red', linestyle='--', label='Worst')
    axes[2].set_xlabel('Chamfer Distance')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Chamfer Distance Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('sample_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'best_rotation': best_rotation_idx,
        'worst_rotation': worst_rotation_idx,
        'best_translation': best_translation_idx,
        'worst_translation': worst_translation_idx,
        'best_chamfer': best_chamfer_idx,
        'worst_chamfer': worst_chamfer_idx,
        'median_rotation': median_rotation_idx,
        'median_translation': median_translation_idx
    }

if __name__ == '__main__':
    recommended_samples = analyze_samples()
    
    print("\n=== 可视化命令建议 ===")
    print("# 查看配准效果最好的样本")
    print(f"python visualize_rpmnet.py --mode progress --sample_idx {recommended_samples['best_rotation']}")
    print(f"python visualize_rpmnet.py --mode 3d --sample_idx {recommended_samples['best_rotation']}")
    
    print("\n# 查看配准效果最差的样本（可能很有趣）")
    print(f"python visualize_rpmnet.py --mode progress --sample_idx {recommended_samples['worst_rotation']}")
    print(f"python visualize_rpmnet.py --mode 3d --sample_idx {recommended_samples['worst_rotation']}")
    
    print("\n# 查看中等表现的样本")
    print(f"python visualize_rpmnet.py --mode progress --sample_idx {recommended_samples['median_rotation']}")
    print(f"python visualize_rpmnet.py --mode 3d --sample_idx {recommended_samples['median_rotation']}")