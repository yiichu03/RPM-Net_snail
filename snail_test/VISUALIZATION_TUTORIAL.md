# 可视化和合成测试教程

本教程介绍如何可视化RPM-Net的迭代过程以及如何创建合成测试数据来精确评估算法性能。

## 目录

1. [方案1: 可视化真实雷达数据的迭代过程](#方案1-可视化真实雷达数据的迭代过程)
2. [方案2: 合成测试数据（已知Ground Truth）](#方案2-合成测试数据已知ground-truth)
3. [完整工作流示例](#完整工作流示例)

---

## 方案1: 可视化真实雷达数据的迭代过程

### 1.1 运行推理并保存可视化数据

首先，使用 `infer_single_pair.py` 处理雷达数据，**必须加上 `--save_vis` 参数**：

```bash
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 3 \
  --num_iter 10 \
  --auto_radius --neighbors 50 \
  --save_vis
```

**生成的文件：**
```
radar_single_frames_original/
├── pred_transforms.npy        # ⭐ 所有迭代的变换矩阵 (1, n_iter, 3, 4)
├── perm_matrices.pickle       # ⭐ 匹配矩阵（稀疏格式）
├── data_dict.npy             # ⭐ 原始点云数据
├── pair_before.ply           # 对齐前可视化
├── pair_after.ply            # 对齐后可视化
└── T_src0_ref3.txt           # 最终变换矩阵
```

### 1.2 可视化迭代过程

#### 选项A: 2D迭代进度图（推荐）

```bash
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress \
  --save registration_progress.png
```

**效果：**
- 显示初始状态 + 每次迭代的结果
- 包含旋转/平移误差
- 2D top view（XY平面）

#### 选项B: 3D可视化

```bash
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode 3d \
  --iteration -1
```

**效果：**
- Open3D交互式3D查看
- 红色：原始source
- 绿色：对齐后的source
- 蓝色：reference
- 可以按'N'显示/隐藏法向量

#### 选项C: 匹配矩阵可视化

```bash
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode matching \
  --iteration -1 \
  --save matching_matrix.png
```

**效果：**
- 显示完整匹配矩阵
- 突出显示强匹配（>95th percentile）
- 统计信息：强匹配数量、平均权重

#### 选项D: 所有迭代的对比图

```bash
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode comparison \
  --save comparison.png
```

**效果：**
- 并排显示所有迭代
- 便于快速比较

### 1.3 解读可视化结果

**好的配准迹象：**
- 绿色点（对齐后source）逐渐与蓝色点（reference）重合
- 旋转/平移误差随迭代减小
- 匹配矩阵中有清晰的对角线或块结构

**问题迹象：**
- 对齐后仍然错位很大
- 误差不收敛或者发散
- 匹配矩阵过于分散（没有强匹配）

---

## 方案2: 合成测试数据（已知Ground Truth）

### 2.1 为什么需要合成测试？

**优势：**
1. **已知ground truth** - 可以精确计算误差
2. **可控变换** - 测试不同旋转/平移幅度
3. **可控partial** - 测试不同遮挡情况
4. **可重复** - 完全确定性

### 2.2 创建基本合成测试

从单个PCD文件创建测试对：

```bash
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_test/ \
  --rotation 30 0 15 \
  --translation 2.0 1.0 0.5 \
  --downsample 1024
```

**参数说明：**
- `--rotation rx ry rz`: 欧拉角（度），绕xyz轴旋转
- `--translation tx ty tz`: 平移（米）
- `--downsample`: 下采样点数

**生成文件：**
```
synthetic_test/
├── synthetic_test.h5                # HDF5格式（compatible with RPM-Net）
├── ground_truth_transform.txt       # Ground truth变换矩阵
├── test_metadata.txt               # 详细元数据
├── source.ply                      # Source点云
├── reference.ply                   # Reference点云
├── pair_ground_truth.ply           # 合并可视化
├── test_files.txt                  # 元数据
└── shape_names.txt                 # 元数据
```

### 2.3 测试Partial Registration

添加partial遮挡来测试算法的鲁棒性：

```bash
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_partial/ \
  --rotation 45 0 0 \
  --translation 3.0 0 0 \
  --downsample 1024 \
  --partial \
  --crop_src right \
  --crop_ref left \
  --crop_ratio 0.3
```

**Partial参数：**
- `--partial`: 启用partial模式
- `--crop_src`: Source裁剪方向（left/right/front/back/top/bottom）
- `--crop_ref`: Reference裁剪方向
- `--crop_ratio`: 裁剪比例（0-1），0.3表示移除30%

**场景示例：**

1. **对向行驶**（source看不到右侧，reference看不到左侧）
   ```bash
   --crop_src right --crop_ref left --crop_ratio 0.3
   ```

2. **前后视角**（source看不到前方，reference看不到后方）
   ```bash
   --crop_src front --crop_ref back --crop_ratio 0.4
   ```

3. **上下遮挡**（模拟不同高度视角）
   ```bash
   --crop_src top --crop_ref bottom --crop_ratio 0.25
   ```

### 2.4 测试合成数据

```bash
python infer_single_pair.py \
  --h5 synthetic_test/synthetic_test.h5 \
  --resume /path/to/checkpoint.pth \
  --src 0 --ref 1 \
  --num_iter 10 \
  --auto_radius --neighbors 40 \
  --save_vis
```

### 2.5 评估精度

**查看ground truth：**
```bash
cat synthetic_test/ground_truth_transform.txt
```

**查看预测结果：**
```bash
cat synthetic_test/T_src0_ref1.txt
```

**计算误差：**
可以手动对比，或者使用Python计算：

```python
import numpy as np

# 加载ground truth和预测
T_gt = np.loadtxt('synthetic_test/ground_truth_transform.txt')
T_pred = np.loadtxt('synthetic_test/T_src0_ref1.txt')

# 计算相对误差
T_error = np.linalg.inv(np.vstack([T_gt, [0,0,0,1]])) @ np.vstack([T_pred, [0,0,0,1]])

# 旋转误差
R_error = T_error[:3, :3]
trace = np.trace(R_error)
rot_error_deg = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi

# 平移误差
trans_error = np.linalg.norm(T_error[:3, 3])

print(f"Rotation error: {rot_error_deg:.3f}°")
print(f"Translation error: {trans_error:.4f}m")
```

---

## 完整工作流示例

### 示例1: 调试真实雷达数据

**目标：** 理解RPM-Net如何处理稀疏雷达数据

```bash
# 1. 运行推理（保存可视化数据）
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 3 \
  --num_iter 10 \
  --auto_radius --neighbors 50 \
  --save_vis

# 2. 可视化迭代过程
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress \
  --save radar_progress.png

# 3. 查看3D结果
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode 3d

# 4. 查看匹配矩阵
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode matching \
  --save matching_matrix.png
```

### 示例2: 基准测试（已知ground truth）

**目标：** 精确测量算法性能

```bash
# 1. 创建合成测试（多个难度）

# 简单：小旋转+小平移
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_easy/ \
  --rotation 15 0 0 \
  --translation 1.0 0.5 0

# 中等：中等旋转+平移
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_medium/ \
  --rotation 45 15 0 \
  --translation 3.0 2.0 0.5

# 困难：大旋转+平移+partial
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_hard/ \
  --rotation 60 30 15 \
  --translation 5.0 3.0 1.0 \
  --partial \
  --crop_src right --crop_ref left \
  --crop_ratio 0.35

# 2. 测试所有案例
for dir in synthetic_easy synthetic_medium synthetic_hard; do
    echo "Testing $dir..."
    python infer_single_pair.py \
      --h5 $dir/synthetic_test.h5 \
      --resume checkpoints/partial-trained.pth \
      --src 0 --ref 1 \
      --num_iter 10 \
      --auto_radius --neighbors 40 \
      --save_vis
    
    # 可视化
    python visualize_radar_registration.py \
      --results_dir $dir/ \
      --mode progress \
      --save $dir/progress.png
done

# 3. 对比ground truth（手动或脚本）
echo "=== Easy Case ==="
echo "Ground Truth:"
cat synthetic_easy/ground_truth_transform.txt
echo "Prediction:"
cat synthetic_easy/T_src0_ref1.txt

echo "=== Medium Case ==="
echo "Ground Truth:"
cat synthetic_medium/ground_truth_transform.txt
echo "Prediction:"
cat synthetic_medium/T_src0_ref1.txt

echo "=== Hard Case ==="
echo "Ground Truth:"
cat synthetic_hard/ground_truth_transform.txt
echo "Prediction:"
cat synthetic_hard/T_src0_ref1.txt
```

### 示例3: 调试partial registration失败

**场景：** 两个partial点云配准失败，想知道原因

```bash
# 1. 创建partial测试
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output debug_partial/ \
  --rotation 30 0 0 \
  --translation 2.0 0 0 \
  --partial \
  --crop_src right --crop_ref left \
  --crop_ratio 0.4

# 2. 先查看ground truth可视化
# 打开 debug_partial/pair_ground_truth.ply 确认overlap

# 3. 运行推理
python infer_single_pair.py \
  --h5 debug_partial/synthetic_test.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 1 \
  --num_iter 15 \
  --auto_radius --neighbors 50 \
  --save_vis

# 4. 可视化迭代过程（看哪一步出错）
python visualize_radar_registration.py \
  --results_dir debug_partial/ \
  --mode progress \
  --save debug_progress.png

# 5. 检查匹配矩阵（看是否找到了对应）
python visualize_radar_registration.py \
  --results_dir debug_partial/ \
  --mode matching \
  --iteration 0  # 检查第一次迭代
  
python visualize_radar_registration.py \
  --results_dir debug_partial/ \
  --mode matching \
  --iteration -1  # 检查最后一次迭代
```

---

## 常见问题

### Q1: `visualize_radar_registration.py` 报错找不到文件

**A:** 确保运行 `infer_single_pair.py` 时加了 `--save_vis` 参数。

### Q2: 可视化图像太小/太模糊

**A:** 使用 `--save` 参数保存高分辨率图像：
```bash
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress \
  --save high_res_progress.png
```
图像会以300 DPI保存，很清晰。

### Q3: 合成测试的ground truth不准确？

**A:** Ground truth是精确的数学变换。如果预测结果与其不符，可能是：
1. 雷达数据too sparse，特征不可靠
2. 参数设置不当（radius, neighbors）
3. Partial overlap不足（尝试减小crop_ratio）

### Q4: 如何测试不同的旋转/平移组合？

**A:** 写一个shell脚本或Python脚本批量测试：

```bash
#!/bin/bash
# test_rotations.sh

for rot in 15 30 45 60; do
    python create_synthetic_test.py \
      --input eagleg7/enhanced/1706001766.376780611.pcd \
      --output synthetic_rot${rot}/ \
      --rotation $rot 0 0 \
      --translation 2.0 0 0
    
    python infer_single_pair.py \
      --h5 synthetic_rot${rot}/synthetic_test.h5 \
      --resume checkpoints/partial-trained.pth \
      --src 0 --ref 1 \
      --auto_radius --neighbors 40 \
      --save_vis
done
```

### Q5: Matching matrix看起来很混乱

**A:** 这可能表明：
1. 特征不够discriminative（稀疏数据问题）
2. Overlap不足
3. 初始对齐太差

尝试：
- 增加 `--neighbors`
- 增加 `--num_iter`
- 查看不同迭代的matching matrix（可能后面会变好）

---

## 总结

**方案1（真实数据可视化）适用于：**
- ✓ 理解算法如何处理真实雷达数据
- ✓ 调试为什么配准失败
- ✓ 展示算法的迭代过程

**方案2（合成测试）适用于：**
- ✓ 精确测量算法性能
- ✓ 测试不同难度场景
- ✓ 验证partial registration能力
- ✓ 基准测试和算法对比

**建议工作流：**
1. 先用方案2（合成测试）验证算法在理想情况下能work
2. 再用方案1（真实数据）调试实际应用中的问题
3. 结合两者理解算法的能力和局限

祝测试顺利！🎯

