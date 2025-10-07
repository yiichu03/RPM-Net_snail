"""Parameter tuning guide for radar registration

This script helps you find the best parameters for your radar data.
"""
import numpy as np
import subprocess
import os
import argparse
from pathlib import Path


class RadarParameterTuner:
    """雷达配准参数调优器"""
    
    def __init__(self, results_dir: str, checkpoint_path: str):
        self.results_dir = Path(results_dir)
        self.checkpoint_path = checkpoint_path
        
    def run_parameter_sweep(self, h5_file: str, src_idx: int = 0, ref_idx: int = 3):
        """运行参数扫描"""
        
        # 参数组合
        param_combinations = [
            # (neighbors, radius, num_iter, description)
            (20, "auto", 5, "Conservative - Small neighborhoods"),
            (30, "auto", 5, "Moderate - Medium neighborhoods"), 
            (50, "auto", 5, "Aggressive - Large neighborhoods"),
            (40, "auto", 10, "More iterations"),
            (60, "auto", 5, "Very large neighborhoods"),
            (30, 2.0, 5, "Fixed radius 2m"),
            (30, 5.0, 5, "Fixed radius 5m"),
            (30, 10.0, 5, "Fixed radius 10m"),
        ]
        
        results = []
        
        for i, (neighbors, radius, num_iter, description) in enumerate(param_combinations):
            print(f"\n=== Testing {i+1}/{len(param_combinations)}: {description} ===")
            print(f"Parameters: neighbors={neighbors}, radius={radius}, num_iter={num_iter}")
            
            # 构建命令
            cmd = [
                "python", "infer_single_pair.py",
                "--h5", h5_file,
                "--resume", self.checkpoint_path,
                "--src", str(src_idx),
                "--ref", str(ref_idx),
                "--num_iter", str(num_iter),
                "--neighbors", str(neighbors),
                "--save_vis"
            ]
            
            if radius == "auto":
                cmd.append("--auto_radius")
            else:
                cmd.extend(["--radius", str(radius)])
            
            # 运行推理
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    # 解析结果
                    final_transform = self._parse_final_transform()
                    if final_transform is not None:
                        rot_error = self._compute_rotation_error(final_transform)
                        trans_error = self._compute_translation_magnitude(final_transform)
                        
                        results.append({
                            'neighbors': neighbors,
                            'radius': radius,
                            'num_iter': num_iter,
                            'description': description,
                            'rotation_error': rot_error,
                            'translation_error': trans_error,
                            'success': True
                        })
                        
                        print(f"✓ Success: Rot={rot_error:.3f}°, Trans={trans_error:.3f}m")
                    else:
                        results.append({
                            'neighbors': neighbors,
                            'radius': radius,
                            'num_iter': num_iter,
                            'description': description,
                            'success': False,
                            'error': 'Failed to parse transform'
                        })
                        print("✗ Failed to parse results")
                else:
                    results.append({
                        'neighbors': neighbors,
                        'radius': radius,
                        'num_iter': num_iter,
                        'description': description,
                        'success': False,
                        'error': result.stderr
                    })
                    print(f"✗ Failed: {result.stderr}")
                    
            except Exception as e:
                results.append({
                    'neighbors': neighbors,
                    'radius': radius,
                    'num_iter': num_iter,
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
                print(f"✗ Exception: {e}")
        
        # 保存结果
        self._save_results(results)
        self._print_summary(results)
        
        return results
    
    def _parse_final_transform(self):
        """解析最终变换矩阵"""
        transform_file = self.results_dir / "T_src0_ref3.txt"
        if not transform_file.exists():
            return None
        
        try:
            transform = np.loadtxt(transform_file)
            return transform
        except:
            return None
    
    def _compute_rotation_error(self, transform: np.ndarray) -> float:
        """计算旋转角度（度）"""
        R = transform[:3, :3]
        trace = np.trace(R)
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        return float(angle_rad * 180 / np.pi)
    
    def _compute_translation_magnitude(self, transform: np.ndarray) -> float:
        """计算平移距离（米）"""
        t = transform[:3, 3]
        return float(np.linalg.norm(t))
    
    def _save_results(self, results):
        """保存结果到文件"""
        results_file = self.results_dir / "parameter_sweep_results.txt"
        
        with open(results_file, 'w') as f:
            f.write("Parameter Sweep Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Test {i+1}: {result['description']}\n")
                f.write(f"  Parameters: neighbors={result['neighbors']}, radius={result['radius']}, num_iter={result['num_iter']}\n")
                
                if result['success']:
                    f.write(f"  ✓ Success: Rot={result['rotation_error']:.3f}°, Trans={result['translation_error']:.3f}m\n")
                else:
                    f.write(f"  ✗ Failed: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
        
        print(f"\nResults saved to: {results_file}")
    
    def _print_summary(self, results):
        """打印结果摘要"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("\n❌ No successful runs!")
            return
        
        print("\n" + "="*60)
        print("PARAMETER SWEEP SUMMARY")
        print("="*60)
        
        # 按旋转误差排序
        successful_results.sort(key=lambda x: x['rotation_error'])
        
        print("\n🏆 Best Results (by rotation error):")
        for i, result in enumerate(successful_results[:3]):
            print(f"  {i+1}. {result['description']}")
            print(f"     Parameters: neighbors={result['neighbors']}, radius={result['radius']}, num_iter={result['num_iter']}")
            print(f"     Errors: Rot={result['rotation_error']:.3f}°, Trans={result['translation_error']:.3f}m")
            print()
        
        # 统计信息
        rot_errors = [r['rotation_error'] for r in successful_results]
        trans_errors = [r['translation_error'] for r in successful_results]
        
        print("📊 Statistics:")
        print(f"  Successful runs: {len(successful_results)}/{len(results)}")
        print(f"  Rotation error: {np.mean(rot_errors):.3f}° ± {np.std(rot_errors):.3f}°")
        print(f"  Translation error: {np.mean(trans_errors):.3f}m ± {np.std(trans_errors):.3f}m")
        print(f"  Best rotation: {np.min(rot_errors):.3f}°")
        print(f"  Best translation: {np.min(trans_errors):.3f}m")
        
        # 推荐参数
        best_result = successful_results[0]
        print(f"\n💡 Recommended parameters:")
        print(f"  --neighbors {best_result['neighbors']}")
        if best_result['radius'] == 'auto':
            print(f"  --auto_radius")
        else:
            print(f"  --radius {best_result['radius']}")
        print(f"  --num_iter {best_result['num_iter']}")


def main():
    parser = argparse.ArgumentParser(description='Parameter Tuning for Radar Registration')
    parser.add_argument('--h5', type=str, required=True,
                       help='HDF5 file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint path')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory')
    parser.add_argument('--src', type=int, default=0,
                       help='Source frame index')
    parser.add_argument('--ref', type=int, default=3,
                       help='Reference frame index')
    
    args = parser.parse_args()
    
    tuner = RadarParameterTuner(args.results_dir, args.checkpoint)
    tuner.run_parameter_sweep(args.h5, args.src, args.ref)


if __name__ == '__main__':
    main()
