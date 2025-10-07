"""Analyze and visualize RPM-Net results

This script provides tools to analyze registration results, compare with ground truth,
and generate detailed evaluation reports.

Usage:
    # Analyze sequence results
    python analyze_results.py \
        --results sequence_results/sequence_results.json \
        --output sequence_results/analysis/

    # Compare with ground truth
    python analyze_results.py \
        --results sequence_results/sequence_results.json \
        --ground_truth ground_truth_poses.txt \
        --output sequence_results/analysis/
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_results(results_path):
    """Load sequence results from JSON"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def load_ground_truth(gt_path):
    """Load ground truth poses
    
    Expected format: Each line is a 3x4 or 4x4 transform matrix (space-separated)
    """
    poses = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Try to read 3 or 4 rows
        rows = []
        for j in range(4):
            if i + j >= len(lines):
                break
            row = lines[i + j].strip()
            if row and not row.startswith('#'):
                values = [float(x) for x in row.split()]
                rows.append(values)
        
        if len(rows) >= 3:
            pose = np.array(rows[:3])  # Take first 3 rows (3x4)
            poses.append(pose)
            i += len(rows)
        else:
            i += 1
    
    return poses


def compute_relative_error(T_pred, T_gt):
    """Compute rotation and translation error between predicted and ground truth"""
    # Convert to 4x4
    if T_pred.shape == (3, 4):
        T_pred = np.vstack([T_pred, [0, 0, 0, 1]])
    if T_gt.shape == (3, 4):
        T_gt = np.vstack([T_gt, [0, 0, 0, 1]])
    
    # Compute relative error
    T_error = np.linalg.inv(T_gt) @ T_pred
    
    # Rotation error (degrees)
    R_error = T_error[:3, :3]
    trace = np.trace(R_error)
    rot_error_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    rot_error_deg = rot_error_rad * 180.0 / np.pi
    
    # Translation error (magnitude)
    t_error = np.linalg.norm(T_error[:3, 3])
    
    return rot_error_deg, t_error


def plot_rotation_translation(results, output_dir):
    """Plot rotation and translation over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    frames = [r['src_frame'] for r in results]
    rotations = [r['rotation_deg'] for r in results]
    translations = [r['translation_m'] for r in results]
    
    ax1.plot(frames, rotations, 'b-o', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Rotation (degrees)')
    ax1.set_title('Rotation per Frame Pair')
    ax1.grid(True)
    
    ax2.plot(frames, translations, 'r-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Translation (meters)')
    ax2.set_title('Translation per Frame Pair')
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'rotation_translation_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()


def plot_inference_time(results, output_dir):
    """Plot inference time distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    times = [r['inference_time_ms'] for r in results]
    frames = [r['src_frame'] for r in results]
    
    ax1.plot(frames, times, 'g-o', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time per Frame Pair')
    ax1.grid(True)
    
    ax2.hist(times, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Inference Time Distribution')
    ax2.axvline(np.mean(times), color='red', linestyle='--', 
                label=f'Mean: {np.mean(times):.1f}ms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'inference_time_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()


def plot_error_comparison(results, gt_poses, output_dir):
    """Plot error comparison with ground truth"""
    errors_rot = []
    errors_trans = []
    frames = []
    
    for result in results:
        src_idx = result['src_frame']
        ref_idx = result['ref_frame']
        
        # Check if ground truth is available
        if src_idx >= len(gt_poses) or ref_idx >= len(gt_poses):
            continue
        
        T_pred = np.array(result['transform'])
        
        # Compute GT relative transform (from src to ref)
        T_src = gt_poses[src_idx]
        T_ref = gt_poses[ref_idx]
        
        # Convert to 4x4
        if T_src.shape == (3, 4):
            T_src = np.vstack([T_src, [0, 0, 0, 1]])
        if T_ref.shape == (3, 4):
            T_ref = np.vstack([T_ref, [0, 0, 0, 1]])
        
        # Relative transform: T_gt = inv(T_src) @ T_ref
        T_gt = np.linalg.inv(T_src) @ T_ref
        T_gt = T_gt[:3, :]
        
        # Compute error
        rot_err, trans_err = compute_relative_error(T_pred, T_gt)
        
        errors_rot.append(rot_err)
        errors_trans.append(trans_err)
        frames.append(src_idx)
    
    if not errors_rot:
        print("Warning: No ground truth comparisons available")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(frames, errors_rot, 'b-o', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Rotation Error (degrees)')
    ax1.set_title('Rotation Error vs Ground Truth')
    ax1.axhline(np.mean(errors_rot), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors_rot):.2f}°')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(frames, errors_trans, 'r-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Translation Error (meters)')
    ax2.set_title('Translation Error vs Ground Truth')
    ax2.axhline(np.mean(errors_trans), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors_trans):.3f}m')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'error_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Error Statistics vs Ground Truth")
    print("="*60)
    print(f"Rotation Error:")
    print(f"  Mean:   {np.mean(errors_rot):6.2f}°")
    print(f"  Median: {np.median(errors_rot):6.2f}°")
    print(f"  Std:    {np.std(errors_rot):6.2f}°")
    print(f"  Min:    {np.min(errors_rot):6.2f}°")
    print(f"  Max:    {np.max(errors_rot):6.2f}°")
    
    print(f"\nTranslation Error:")
    print(f"  Mean:   {np.mean(errors_trans):6.3f}m")
    print(f"  Median: {np.median(errors_trans):6.3f}m")
    print(f"  Std:    {np.std(errors_trans):6.3f}m")
    print(f"  Min:    {np.min(errors_trans):6.3f}m")
    print(f"  Max:    {np.max(errors_trans):6.3f}m")


def generate_summary_report(results, gt_poses, output_dir):
    """Generate comprehensive summary report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RPM-Net Sequence Processing Summary Report\n")
        f.write("="*60 + "\n\n")
        
        # Basic info
        f.write(f"Total frame pairs processed: {len(results)}\n")
        f.write(f"Frame range: {results[0]['src_frame']} to {results[-1]['ref_frame']}\n\n")
        
        # Rotation statistics
        rotations = [r['rotation_deg'] for r in results]
        f.write("Rotation Statistics:\n")
        f.write(f"  Mean:   {np.mean(rotations):6.2f}°\n")
        f.write(f"  Median: {np.median(rotations):6.2f}°\n")
        f.write(f"  Std:    {np.std(rotations):6.2f}°\n")
        f.write(f"  Min:    {np.min(rotations):6.2f}°\n")
        f.write(f"  Max:    {np.max(rotations):6.2f}°\n\n")
        
        # Translation statistics
        translations = [r['translation_m'] for r in results]
        f.write("Translation Statistics:\n")
        f.write(f"  Mean:   {np.mean(translations):6.3f}m\n")
        f.write(f"  Median: {np.median(translations):6.3f}m\n")
        f.write(f"  Std:    {np.std(translations):6.3f}m\n")
        f.write(f"  Min:    {np.min(translations):6.3f}m\n")
        f.write(f"  Max:    {np.max(translations):6.3f}m\n\n")
        
        # Inference time
        times = [r['inference_time_ms'] for r in results]
        f.write("Inference Time Statistics:\n")
        f.write(f"  Mean:   {np.mean(times):6.1f}ms\n")
        f.write(f"  Median: {np.median(times):6.1f}ms\n")
        f.write(f"  Std:    {np.std(times):6.1f}ms\n")
        f.write(f"  Min:    {np.min(times):6.1f}ms\n")
        f.write(f"  Max:    {np.max(times):6.1f}ms\n\n")
        
        # Ground truth comparison if available
        if gt_poses is not None:
            errors_rot = []
            errors_trans = []
            
            for result in results:
                src_idx = result['src_frame']
                ref_idx = result['ref_frame']
                
                if src_idx >= len(gt_poses) or ref_idx >= len(gt_poses):
                    continue
                
                T_pred = np.array(result['transform'])
                T_src = gt_poses[src_idx]
                T_ref = gt_poses[ref_idx]
                
                if T_src.shape == (3, 4):
                    T_src = np.vstack([T_src, [0, 0, 0, 1]])
                if T_ref.shape == (3, 4):
                    T_ref = np.vstack([T_ref, [0, 0, 0, 1]])
                
                T_gt = np.linalg.inv(T_src) @ T_ref
                T_gt = T_gt[:3, :]
                
                rot_err, trans_err = compute_relative_error(T_pred, T_gt)
                errors_rot.append(rot_err)
                errors_trans.append(trans_err)
            
            if errors_rot:
                f.write("="*60 + "\n")
                f.write("Ground Truth Comparison\n")
                f.write("="*60 + "\n\n")
                
                f.write("Rotation Error:\n")
                f.write(f"  Mean:   {np.mean(errors_rot):6.2f}°\n")
                f.write(f"  Median: {np.median(errors_rot):6.2f}°\n")
                f.write(f"  RMSE:   {np.sqrt(np.mean(np.array(errors_rot)**2)):6.2f}°\n")
                f.write(f"  Max:    {np.max(errors_rot):6.2f}°\n\n")
                
                f.write("Translation Error:\n")
                f.write(f"  Mean:   {np.mean(errors_trans):6.3f}m\n")
                f.write(f"  Median: {np.median(errors_trans):6.3f}m\n")
                f.write(f"  RMSE:   {np.sqrt(np.mean(np.array(errors_trans)**2)):6.3f}m\n")
                f.write(f"  Max:    {np.max(errors_trans):6.3f}m\n\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RPM-Net sequence results")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to sequence_results.json')
    parser.add_argument('--output', type=str, default='analysis/',
                       help='Output directory for analysis')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth poses (optional)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} frame pair results")
    
    # Load ground truth if provided
    gt_poses = None
    if args.ground_truth:
        print(f"Loading ground truth from: {args.ground_truth}")
        gt_poses = load_ground_truth(args.ground_truth)
        print(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    plot_rotation_translation(results, output_dir)
    plot_inference_time(results, output_dir)
    
    if gt_poses:
        plot_error_comparison(results, gt_poses, output_dir)
    
    # Generate summary report
    generate_summary_report(results, gt_poses, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

