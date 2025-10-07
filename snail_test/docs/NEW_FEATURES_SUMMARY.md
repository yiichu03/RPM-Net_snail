# 新功能总结：可视化和合成测试

本文档总结了为解决"难以肉眼判断配准效果"和"需要已知ground truth测试"而添加的新功能。

## 🎯 解决的问题

### 问题1: 难以判断迭代效果
**你的原话：** "我感觉我肉眼很难看出来infer_single_pair对比的情况"

**解决方案：** 
- ✅ 保存所有迭代的变换矩阵
- ✅ 保存匹配矩阵（permutation matrices）
- ✅ 创建可视化工具显示迭代进度

### 问题2: 需要已知ground truth测试
**你的原话：** "使用同一个pcd文件的点云作为source，然后我们手动给他一个旋转和平移得到target点云"

**解决方案：**
- ✅ 创建合成测试脚本
- ✅ 支持手动设置旋转和平移
- ✅ 支持partial遮挡（不同区域）
- ✅ 提供精确的ground truth

---

## 📦 新增文件

### 1. 修改的文件

#### `infer_single_pair.py` ⭐
**新增功能：** 保存eval格式的结果文件

**新增输出：**
```
radar_single_frames_original/
├── pred_transforms.npy        # (1, n_iter, 3, 4) 所有迭代的变换
├── perm_matrices.pickle       # 匹配矩阵（稀疏格式）
├── data_dict.npy             # 原始点云数据
└── ... (原有的PLY文件等)
```

**改动：**
- 第177-209行：保存详细结果用于可视化
- 第228-237行：保存数据字典

### 2. 新增文件

#### `visualize_radar_registration.py` (358行)
**功能：** 可视化雷达数据的配准过程

**4种可视化模式：**
1. **progress** - 显示迭代进度（2D）
2. **3d** - 3D交互式可视化
3. **matching** - 显示匹配矩阵
4. **comparison** - 并排对比所有迭代

**使用示例：**
```bash
# 查看迭代进度
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress \
  --save progress.png

# 3D可视化
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode 3d

# 查看匹配矩阵
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode matching \
  --save matching.png
```

#### `create_synthetic_test.py` (421行)
**功能：** 创建已知ground truth的合成测试数据

**支持功能：**
- ✅ 从单个PCD创建source和target
- ✅ 自定义旋转（欧拉角，度）
- ✅ 自定义平移（米）
- ✅ Partial遮挡（6个方向：left/right/front/back/top/bottom）
- ✅ 不同区域遮挡（测试partial registration）
- ✅ 自动生成ground truth

**使用示例：**
```bash
# 基本测试
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_test/ \
  --rotation 30 0 15 \
  --translation 2.0 1.0 0.5

# Partial测试（source右侧遮挡，reference左侧遮挡）
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_partial/ \
  --rotation 45 0 0 \
  --translation 3.0 0 0 \
  --partial \
  --crop_src right \
  --crop_ref left \
  --crop_ratio 0.3
```

#### `VISUALIZATION_TUTORIAL.md` (588行)
**功能：** 完整的使用教程

**内容包括：**
- 方案1：可视化真实雷达数据
- 方案2：合成测试数据
- 完整工作流示例
- 常见问题解答

---

## 🚀 完整工作流

### 工作流1：调试真实雷达数据

```bash
# 1. 运行推理（必须加--save_vis！）
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

# 4. 查看匹配矩阵（理解对应关系）
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode matching \
  --save matching_matrix.png
```

**输出：**
- `radar_progress.png` - 看到每次迭代如何改善对齐
- 3D窗口 - 交互式查看结果
- `matching_matrix.png` - 看到哪些点匹配了

### 工作流2：精确性能测试

```bash
# 1. 创建合成测试（多个难度）
# 简单
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_easy/ \
  --rotation 15 0 0 \
  --translation 1.0 0.5 0

# 困难（大旋转+平移+partial）
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_hard/ \
  --rotation 60 30 15 \
  --translation 5.0 3.0 1.0 \
  --partial \
  --crop_src right --crop_ref left \
  --crop_ratio 0.35

# 2. 测试
python infer_single_pair.py \
  --h5 synthetic_hard/synthetic_test.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 1 \
  --num_iter 10 \
  --auto_radius --neighbors 40 \
  --save_vis

# 3. 对比ground truth
echo "Ground Truth:"
cat synthetic_hard/ground_truth_transform.txt
echo "Prediction:"
cat synthetic_hard/T_src0_ref1.txt

# 4. 可视化迭代过程
python visualize_radar_registration.py \
  --results_dir synthetic_hard/ \
  --mode progress \
  --save synthetic_progress.png
```

**好处：**
- ✅ 精确知道误差有多大
- ✅ 测试极限情况
- ✅ 验证partial功能

---

## 📊 输出示例

### 1. 迭代进度图（progress mode）

```
+-------------------+-------------------+-------------------+
| Initial           | Iteration 1       | Iteration 2       |
| (Before)          | Rot: 28.5°        | Rot: 12.3°        |
|                   | Trans: 1.8m       | Trans: 0.8m       |
| Red: Source       |                   |                   |
| Blue: Reference   | Green: Aligned    | Green: Aligned    |
|                   | Blue: Reference   | Blue: Reference   |
+-------------------+-------------------+-------------------+
| Iteration 3       | Iteration 4       | Iteration 5       |
| Rot: 5.2°         | Rot: 2.1°         | Rot: 0.8°         |
| Trans: 0.3m       | Trans: 0.1m       | Trans: 0.05m      |
+-------------------+-------------------+-------------------+
```

**解读：**
- 绿色点逐渐靠近蓝色点 ✓
- 误差逐渐减小 ✓
- 说明收敛良好

### 2. 匹配矩阵（matching mode）

```
Full Matrix                Strong Matches (>95th percentile)
[显示热力图]                [显示稀疏矩阵]

Strong matches: 156 | Avg weight: 0.0234 | Threshold: 0.0187
```

**解读：**
- 强匹配数量：越多越好（说明找到了对应）
- 如果矩阵很分散：可能overlap不足或特征不可靠

### 3. 合成测试的元数据

```
Synthetic Test Metadata
============================================================

Input PCD: eagleg7/enhanced/1706001766.376780611.pcd
Original points: 1523
Downsampled points: 1024

Ground Truth Transform:
  Rotation (xyz): (45.0, 0.0, 0.0) degrees
  Translation: (3.0, 0.0, 0.0) meters

Partial Settings:
  Source crop: right (30.0%) -> 717 points
  Reference crop: left (30.0%) -> 721 points
  Overlap estimate: ~40%

Normal estimation: k=30
```

---

## 🔍 故障排查

### 问题1: `visualize_radar_registration.py` 报错找不到文件

**原因：** 没有运行 `infer_single_pair.py` 时加 `--save_vis`

**解决：** 重新运行推理，确保加上 `--save_vis`

### 问题2: 迭代过程看起来不收敛

**可能原因：**
1. Radius太小 → 增加 `--radius` 或使用 `--auto_radius`
2. Neighbors太少 → 增加 `--neighbors` (40-50)
3. 数据太sparse → 考虑accumulate多帧
4. Overlap不足 → 选择更接近的帧

**调试步骤：**
1. 查看匹配矩阵（是否有强匹配）
2. 查看3D可视化（手动评估overlap）
3. 尝试合成测试（排除数据问题）

### 问题3: 合成测试也失败

**说明：** 可能是：
1. Partial overlap太小 → 减小 `--crop_ratio`
2. 旋转/平移太大 → 减小幅度
3. 模型参数不合适 → 调整radius/neighbors

**建议：**
从简单case开始（小旋转+小平移，no partial），逐步增加难度

---

## 💡 关键见解

### 你的想法1：可视化迭代过程 ✅
**实现：**
- `infer_single_pair.py` 保存所有迭代数据
- `visualize_radar_registration.py` 提供4种可视化模式
- 可以清楚看到每次迭代的改进

### 你的想法2：合成测试（已知ground truth）✅
**实现：**
- `create_synthetic_test.py` 从单个PCD创建测试对
- 支持任意旋转/平移组合
- 支持不同区域的partial遮挡
- 自动生成ground truth用于精确评估

### Bonus：匹配矩阵可视化 🎁
**额外功能：**
- 可视化Sinkhorn算法的输出
- 理解哪些点被匹配了
- 诊断配准失败的原因

---

## 📚 文档索引

1. **`VISUALIZATION_TUTORIAL.md`** - 完整使用教程
   - 方案1：真实数据可视化
   - 方案2：合成测试
   - 完整工作流示例
   - 常见问题

2. **`NEW_FEATURES_SUMMARY.md`** (本文档) - 功能总结
   - 新增文件说明
   - 快速参考
   - 故障排查

3. **脚本内注释** - 详细参数说明
   - `visualize_radar_registration.py` - 每个模式的用法
   - `create_synthetic_test.py` - 所有参数的含义

---

## ✅ 检查清单

在使用新功能前，确保：

- [ ] 运行 `infer_single_pair.py` 时加了 `--save_vis`
- [ ] 生成了 `pred_transforms.npy` 和 `perm_matrices.pickle`
- [ ] 安装了所有依赖（matplotlib, scipy, pickle）
- [ ] 阅读了 `VISUALIZATION_TUTORIAL.md`

---

## 🎯 推荐使用顺序

### 第一次使用：
1. 创建简单的合成测试（验证工具work）
2. 可视化合成测试结果（熟悉可视化工具）
3. 应用到真实雷达数据

### 日常调试：
1. 运行推理（记得加 `--save_vis`）
2. 可视化progress mode（快速查看效果）
3. 如果有问题，查看matching matrix（诊断原因）

### 性能评估：
1. 创建多个难度的合成测试
2. 批量测试
3. 对比ground truth
4. 生成报告

---

希望这些工具能帮你更好地理解和调试RPM-Net在雷达数据上的表现！🚀

如有问题，参考 `VISUALIZATION_TUTORIAL.md` 的常见问题部分。

