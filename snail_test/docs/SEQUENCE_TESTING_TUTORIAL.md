# 序列测试教程 - 获取轨迹估计

本教程详细说明如何使用RPM-Net处理雷达帧序列来估计轨迹。

## 目录

1. [前提条件](#1-前提条件)
2. [单帧对测试（验证）](#2-单帧对测试验证)
3. [序列处理（获取轨迹）](#3-序列处理获取轨迹)
4. [结果分析](#4-结果分析)
5. [可变点数支持](#5-可变点数支持)

---

## 1. 前提条件

### 1.1 确保单帧对测试成功

在进行序列测试之前，**务必先验证单帧对配准是否工作**：

```bash
cd snail_test

# 测试单帧对
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius --neighbors 30 --save_vis
```

**检查结果：**
- 打开 `pair_after.ply`，验证点云是否对齐
- 旋转误差应该 <15°，平移误差应该 <2m
- 如果不满足，请先调整参数（见QUICKSTART.md）

### 1.2 准备序列数据

**选项A: 使用固定点数（下采样到1024）**

```bash
python convert_radar_single_frames.py
```

默认配置会将所有帧下采样到1024点。

**选项B: 使用原始点数（保留变化的点数）**

编辑 `convert_radar_single_frames.py` 底部：

```python
convert_single_radar_frames( 
    radar_dir="./eagleg7/enhanced/",
    output_dir="radar_single_frames_original/",
    frame_indices=list(range(0, 50, 2)),  # 每隔2帧取一帧，共25帧
    downsample_points=1024,  # 这个参数会被忽略
    k_normal=30,
    log_filename="normals_log.txt",
    keep_original_points=True  # 保留原始点数
)
```

然后运行：

```bash
python convert_radar_single_frames.py
```

---

## 2. 单帧对测试（验证）

### 2.1 基本测试

测试相邻帧（0→1）：

```bash
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius --neighbors 30 --save_vis
```

### 2.2 测试较大运动

测试间隔更大的帧（0→10）：

```bash
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 10 \
  --auto_radius --neighbors 40 --num_iter 10 --save_vis
```

**注意：** 间隔越大，配准越困难，可能需要：
- 增加 `--neighbors` (40-50)
- 增加 `--num_iter` (10-15)
- 手动设置更大的 `--radius` (5-7m)

---

## 3. 序列处理（获取轨迹）

现在我们已经创建了 `process_sequence.py` 来自动处理整个序列！

### 3.1 处理所有相邻帧对

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

**这个命令会：**
1. 处理所有相邻帧对：(0→1), (1→2), (2→3), ...
2. 累积变换得到完整轨迹
3. 保存详细结果到JSON文件
4. 生成轨迹可视化图

### 3.2 使用步长（跳帧处理）

如果帧数很多，可以每隔N帧处理一次：

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results_stride5/ \
  --stride 5 \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

**stride=5 意味着：** 处理 (0→5), (5→10), (10→15), ...

**优点：**
- 处理速度快
- 累积误差少（因为变换次数少）

**缺点：**
- 帧间运动大，配准可能失败
- 需要调整参数（增加neighbors, num_iter）

### 3.3 处理特定帧范围

只处理特定范围的帧：

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results_subset/ \
  --start_frame 10 \
  --end_frame 30 \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

### 3.4 详细模式（调试用）

查看每一对的详细结果：

```bash
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory \
  --verbose
```

**verbose模式会打印：**
```
Pair (  0 →   1): rot=  3.45°, trans= 0.123m, time= 78.5ms
Pair (  1 →   2): rot=  2.89°, trans= 0.098m, time= 81.2ms
...
```

---

## 4. 结果分析

### 4.1 查看输出文件

处理完成后，`sequence_results/` 目录包含：

```
sequence_results/
├── sequence_results.json        # 每对帧的详细结果
├── trajectory.npy              # 累积的3D轨迹位置
├── pairwise_transforms.npy     # 所有帧对的变换矩阵
└── trajectory_visualization.png # 轨迹4视图可视化
```

### 4.2 查看轨迹可视化

打开 `trajectory_visualization.png`，包含4个视图：
- **3D View**: 完整三维轨迹
- **Top View (XY)**: 俯视图（最常用）
- **Side View (XZ)**: 侧视图
- **Front View (YZ)**: 正视图

### 4.3 分析JSON结果

```bash
# 在Linux/Mac上
cat sequence_results/sequence_results.json | head -50

# 在Windows上
type sequence_results\sequence_results.json | more
```

每对帧的结果包括：
```json
{
  "src_frame": 0,
  "ref_frame": 1,
  "transform": [[R11, R12, R13, tx], [R21, R22, R23, ty], [R31, R32, R33, tz]],
  "rotation_deg": 3.45,
  "translation_m": 0.123,
  "inference_time_ms": 78.5,
  "accumulated_position": [0.1, 0.05, 0.0]
}
```

### 4.4 使用分析脚本

运行完整的统计分析：

```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --output sequence_results/analysis/
```

**生成的文件：**
- `summary_report.txt`: 统计摘要（旋转、平移、时间）
- `rotation_translation_plot.png`: 每帧的旋转/平移曲线
- `inference_time_plot.png`: 推理时间分布

**查看摘要报告：**
```bash
cat sequence_results/analysis/summary_report.txt
```

---

## 5. 可变点数支持

### 5.1 为什么使用可变点数？

**优点：**
- 保留原始雷达返回的所有信息
- 不丢失稀疏数据中的任何点
- 更真实地反映雷达特性

**缺点：**
- HDF5文件格式稍微复杂一些
- 每帧点数可能差异很大

### 5.2 转换可变点数数据

```python
# 编辑 convert_radar_single_frames.py
convert_single_radar_frames( 
    radar_dir="./eagleg7/enhanced/",
    output_dir="radar_single_frames_original/",
    frame_indices=[0, 5, 10, 15, 20],
    downsample_points=1024,  # 这个参数会被忽略
    k_normal=30,
    log_filename="normals_log.txt",
    keep_original_points=True  # 关键：保留原始点数
)
```

### 5.3 验证可变点数

```bash
python validate_converted_data.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5
```

**应该看到：**
```
Loaded variable-size frames:
  Source frame 0: 347 points
  Reference frame 1: 521 points
```

### 5.4 使用可变点数进行推理

**完全相同的命令！** 脚本会自动检测：

```bash
# 单帧对
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius --neighbors 30 --save_vis

# 序列处理
python process_sequence.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results_original/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

**RPM-Net原生支持不同点数！** 
- 源点云可以是 (B, J, 6)
- 目标点云可以是 (B, K, 6)
- J 和 K 可以不同！

---

## 6. 完整工作流示例

### 示例1: 快速测试（3帧）

```bash
# 1. 转换3帧（默认配置）
python convert_radar_single_frames.py

# 2. 验证
python validate_converted_data.py --h5 radar_single_frames/radar_single_frames_test0.h5

# 3. 单帧对测试
python infer_single_pair.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius --neighbors 30 --save_vis

# 4. 检查pair_after.ply，如果OK，继续
# 5. 序列处理
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory

# 6. 查看结果
# 打开 sequence_results/trajectory_visualization.png
```

### 示例2: 完整序列（50帧，stride=2）

```bash
# 1. 修改 convert_radar_single_frames.py
# 设置 frame_indices=list(range(0, 100, 2))  # 50帧
python convert_radar_single_frames.py

# 2. 处理相邻帧
python process_sequence.py \
  --h5 radar_single_frames/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results_full/ \
  --stride 1 \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory \
  --verbose

# 3. 分析
python analyze_results.py \
  --results sequence_results_full/sequence_results.json \
  --output sequence_results_full/analysis/

# 4. 查看摘要
cat sequence_results_full/analysis/summary_report.txt
```

### 示例3: 使用原始点数（更真实）

```bash
# 1. 转换（保留原始点数）
# 修改 convert_radar_single_frames.py，设置 keep_original_points=True
python convert_radar_single_frames.py

# 2. 检查帧的映射
cat radar_single_frames_original/frame_indices_mapping.txt

# 3. 验证（查看每帧点数）
python validate_converted_data.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5

# 4. 单帧对测试
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --auto_radius --neighbors 30 --save_vis

# 5. 序列处理（完全相同的命令！）
python process_sequence.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --output_dir sequence_results_original/ \
  --auto_radius \
  --neighbors 30 \
  --visualize_trajectory
```

---

## 7. 故障排除

### 问题1: 序列中某些帧对失败

**症状：** 某些帧的旋转误差很大（>30°）

**解决方案：**
1. 检查这些帧的重叠度
   ```bash
   python validate_converted_data.py \
     --h5 radar_single_frames/radar_single_frames_test0.h5 \
     --check_overlap X Y
   ```
2. 如果重叠<30%，增加stride或跳过这些帧
3. 尝试增加 `--neighbors` 和 `--num_iter`

### 问题2: 轨迹漂移

**症状：** 轨迹随时间严重偏离

**这是正常的！** 序列里程计会累积误差：
- 每帧的小误差会累积
- 10帧后可能有明显漂移
- 50帧后可能漂移很大

**改进方法：**
1. 使用更大的stride（减少累积次数）
2. 实现闭环检测（高级）
3. 与其他传感器融合（IMU、轮式里程计）

### 问题3: 处理速度慢

**加速方法：**
1. 使用GPU：`--gpu 0`
2. 增加stride：`--stride 5`
3. 减少帧数：`--start_frame X --end_frame Y`
4. 减少迭代次数：`--num_iter 3`（但可能降低精度）

### 问题4: 内存不足

**解决方案：**
1. 减少一次处理的帧数
2. 使用更大的stride
3. 分批处理：
   ```bash
   # 处理0-20帧
   python process_sequence.py ... --start_frame 0 --end_frame 20
   # 处理20-40帧
   python process_sequence.py ... --start_frame 20 --end_frame 40
   ```

---

## 8. 进阶技巧

### 8.1 与真值对比

如果有GPS/SLAM真值轨迹：

```bash
python analyze_results.py \
  --results sequence_results/sequence_results.json \
  --ground_truth ground_truth_poses.txt \
  --output sequence_results/analysis/
```

**ground_truth_poses.txt 格式：**
```
# 每个变换矩阵占3行或4行（3x4或4x4）
R11 R12 R13 tx
R21 R22 R23 ty
R31 R32 R33 tz

R11 R12 R13 tx
R21 R22 R23 ty
R31 R32 R33 tz
...
```

### 8.2 查看帧的映射关系

转换后会生成映射文件：

```bash
cat radar_single_frames/frame_indices_mapping.txt
```

输出：
```
# Frame Index -> PCD Filename Mapping
# Total frames selected: 3
# Original frame indices: [0, 10, 20]

0       1706001766.376780611.pcd
10      1706001767.876780611.pcd
20      1706001769.376780611.pcd
```

### 8.3 可视化中间迭代

如果想看RPM-Net如何逐步优化：

在 `infer_single_pair.py` 中，`transforms` 列表包含每次迭代的结果：

```python
# 在第188行附近添加
for iter_i, T in enumerate(transforms):
    T_np = T[0].detach().cpu().numpy()
    xyz_aligned = apply_se3(xyz_src, T_np)
    # 保存中间结果
    pcd_iter = o3d.geometry.PointCloud(...)
    o3d.io.write_point_cloud(f"iter_{iter_i}.ply", pcd_iter)
```

---

## 9. 性能基准

### 预期性能（稀疏雷达）

| 场景 | 旋转误差 | 平移误差 | 推理时间 |
|------|----------|----------|----------|
| 相邻帧（小运动） | <5° | <0.5m | 50-100ms |
| 间隔5帧（中等运动） | <10° | <1.0m | 100-200ms |
| 间隔10帧（大运动） | <20° | <2.0m | 150-300ms |

### 与ModelNet40对比

| 指标 | ModelNet40 | 雷达 |
|------|-----------|------|
| 旋转误差 | <1° | <5° |
| 平移误差 | <0.05m | <0.5m |
| 数据密度 | 密集 | 极稀疏 |
| 场景尺度 | ~1m | ~50m |

---

## 10. 总结

**关键步骤回顾：**

1. ✅ 转换数据：`convert_radar_single_frames.py`
   - 可选：固定点数或原始点数
   - 会生成frame_indices_mapping.txt

2. ✅ 验证数据：`validate_converted_data.py`
   - 检查法向量质量
   - 确认重叠度

3. ✅ 单帧对测试：`infer_single_pair.py`
   - 验证配准工作
   - 调整参数

4. ✅ 序列处理：`process_sequence.py` ⭐ **新工具！**
   - 自动处理所有帧对
   - 累积变换得到轨迹
   - 生成可视化

5. ✅ 分析结果：`analyze_results.py`
   - 统计摘要
   - 可视化曲线
   - 与真值对比（可选）

**你的代码已经完全支持序列测试和轨迹估计！** 🎉

有任何问题，请参考 `TESTING_GUIDE.md` 或 `README.md`。

