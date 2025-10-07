# 快速参考：可视化和合成测试

## 🎯 两个核心功能

### 方案1: 可视化迭代过程（真实数据）
**目的：** 看懂RPM-Net如何一步步对齐点云

```bash
# Step 1: 运行推理（⚠️ 必须加 --save_vis！）
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 3 \
  --num_iter 10 \
  --auto_radius --neighbors 50 \
  --save_vis

# Step 2: 可视化迭代进度
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress \
  --save progress.png
```

### 方案2: 合成测试（已知ground truth）
**目的：** 精确测试算法性能

```bash
# Step 1: 创建测试数据
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output synthetic_test/ \
  --rotation 30 0 15 \
  --translation 2.0 1.0 0.5

# Step 2: 测试
python infer_single_pair.py \
  --h5 synthetic_test/synthetic_test.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 1 \
  --num_iter 10 \
  --auto_radius --neighbors 40 \
  --save_vis

# Step 3: 对比ground truth
cat synthetic_test/ground_truth_transform.txt  # Ground truth
cat synthetic_test/T_src0_ref1.txt            # Prediction
```

---

## 📊 可视化模式速查

```bash
# 1. 迭代进度图（最常用）
python visualize_radar_registration.py \
  --results_dir RESULTS_DIR/ \
  --mode progress \
  --save progress.png

# 2. 3D交互式查看
python visualize_radar_registration.py \
  --results_dir RESULTS_DIR/ \
  --mode 3d

# 3. 匹配矩阵
python visualize_radar_registration.py \
  --results_dir RESULTS_DIR/ \
  --mode matching \
  --save matching.png

# 4. 并排对比所有迭代
python visualize_radar_registration.py \
  --results_dir RESULTS_DIR/ \
  --mode comparison \
  --save comparison.png
```

---

## 🔧 合成测试参数速查

### 基本用法
```bash
python create_synthetic_test.py \
  --input PCD_FILE \
  --output OUTPUT_DIR/ \
  --rotation RX RY RZ \      # 欧拉角（度）
  --translation TX TY TZ      # 平移（米）
```

### Partial遮挡
```bash
python create_synthetic_test.py \
  --input PCD_FILE \
  --output OUTPUT_DIR/ \
  --rotation 45 0 0 \
  --translation 3.0 0 0 \
  --partial \
  --crop_src right \          # Source裁剪方向
  --crop_ref left \           # Reference裁剪方向
  --crop_ratio 0.3            # 裁剪30%
```

**裁剪方向：** `left` `right` `front` `back` `top` `bottom`

---

## ⚠️ 常见错误

### 错误1: "FileNotFoundError: data_dict.npy"
**原因：** 没加 `--save_vis`  
**解决：** 重新运行 `infer_single_pair.py --save_vis`

### 错误2: 可视化图像看不清
**解决：** 使用 `--save` 参数保存高分辨率图像
```bash
python visualize_radar_registration.py ... --save output.png
```

### 错误3: 迭代不收敛
**调试步骤：**
1. 查看匹配矩阵：`--mode matching`
2. 增加neighbors：`--neighbors 50`
3. 尝试合成测试排除数据问题

---

## 📁 输出文件说明

### 推理输出（加 --save_vis 后）
```
radar_single_frames_original/
├── pred_transforms.npy        # ⭐ 所有迭代变换 (1, n_iter, 3, 4)
├── perm_matrices.pickle       # ⭐ 匹配矩阵（用于可视化）
├── data_dict.npy             # ⭐ 原始点云
├── pair_before.ply           # 对齐前
├── pair_after.ply            # 对齐后
└── T_src0_ref3.txt           # 最终变换
```

### 合成测试输出
```
synthetic_test/
├── synthetic_test.h5              # HDF5数据
├── ground_truth_transform.txt     # ⭐ Ground truth
├── source.ply                    # Source点云
├── reference.ply                 # Reference点云
├── pair_ground_truth.ply         # 合并可视化
└── test_metadata.txt             # 元数据
```

---

## 🚀 典型工作流

### 场景1: 调试真实雷达配准

```bash
# 1. 推理
python infer_single_pair.py \
  --h5 radar_single_frames_original/radar_single_frames_test0.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 3 --num_iter 10 \
  --auto_radius --neighbors 50 --save_vis

# 2. 看迭代过程
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode progress

# 3. 看匹配情况
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode matching

# 4. 3D查看
python visualize_radar_registration.py \
  --results_dir radar_single_frames_original/ \
  --mode 3d
```

### 场景2: 测试Partial能力

```bash
# 1. 创建partial测试（source右侧遮挡，reference左侧遮挡）
python create_synthetic_test.py \
  --input eagleg7/enhanced/1706001766.376780611.pcd \
  --output partial_test/ \
  --rotation 45 0 0 --translation 3.0 0 0 \
  --partial --crop_src right --crop_ref left --crop_ratio 0.3

# 2. 测试
python infer_single_pair.py \
  --h5 partial_test/synthetic_test.h5 \
  --resume checkpoints/partial-trained.pth \
  --src 0 --ref 1 --num_iter 10 \
  --auto_radius --neighbors 40 --save_vis

# 3. 查看结果
python visualize_radar_registration.py \
  --results_dir partial_test/ \
  --mode progress --save partial_progress.png

# 4. 对比ground truth
cat partial_test/ground_truth_transform.txt
cat partial_test/T_src0_ref1.txt
```

### 场景3: 批量性能测试

```bash
# 创建多个难度的测试
for rot in 15 30 45 60; do
    python create_synthetic_test.py \
      --input eagleg7/enhanced/1706001766.376780611.pcd \
      --output synthetic_rot${rot}/ \
      --rotation $rot 0 0 --translation 2.0 0 0
    
    python infer_single_pair.py \
      --h5 synthetic_rot${rot}/synthetic_test.h5 \
      --resume checkpoints/partial-trained.pth \
      --src 0 --ref 1 --num_iter 10 \
      --auto_radius --neighbors 40 --save_vis
    
    python visualize_radar_registration.py \
      --results_dir synthetic_rot${rot}/ \
      --mode progress --save rot${rot}_progress.png
done
```

---

## 📖 完整文档

- **详细教程：** `VISUALIZATION_TUTORIAL.md`
- **功能总结：** `NEW_FEATURES_SUMMARY.md`
- **本文档：** 快速参考和常用命令

---

## 💡 核心要点

1. **⚠️ 推理时必须加 `--save_vis`** 才能可视化
2. **方案1（真实数据）** 用于理解和调试
3. **方案2（合成测试）** 用于精确评估
4. **4种可视化模式** 各有用途，progress最常用
5. **Partial测试** 通过不同方向裁剪实现

---

需要帮助？查看完整教程：`VISUALIZATION_TUTORIAL.md`

