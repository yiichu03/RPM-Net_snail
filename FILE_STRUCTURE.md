# RPM-Net 雷达测试项目文件结构

## 📁 项目根目录

```
RPMNet/
├── src/                          # RPM-Net 原始代码
├── snail_test/                   # 雷达测试相关代码和数据
├── datasets/                     # 数据集目录
├── eval_results/                 # 评估结果
├── checkpoints/                  # 模型检查点
├── docs/                         # 文档
├── .gitignore                    # Git忽略文件
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明
└── LICENSE                       # 许可证
```

---

## 🔧 核心代码目录 (`src/`)

### 原始 RPM-Net 代码

#### 模型相关
- **`src/models/rpmnet.py`** - RPM-Net 核心模型实现
  ```bash
  # 这是RPM-Net的主要模型文件，包含迭代配准算法
  ```

- **`src/models/feature_nets.py`** - 特征提取网络
  ```bash
  # 实现PPF特征提取，用于点云配准
  ```

- **`src/models/pointnet_util.py`** - PointNet工具函数
  ```bash
  # PointNet相关的工具函数
  ```

#### 数据处理
- **`src/data_loader/datasets.py`** - 数据集加载器
  ```bash
  # 加载HDF5格式的ModelNet40数据
  ```

- **`src/data_loader/transforms.py`** - 数据变换
  ```bash
  # 数据预处理和增强变换
  ```

#### 训练和评估
- **`src/train.py`** - 训练脚本
  ```bash
  python src/train.py --dataset modelnet40 --batch_size 32
  ```

- **`src/eval.py`** - 评估脚本
  ```bash
  python src/eval.py --dataset modelnet40 --resume checkpoints/model.pth
  ```

#### 工具和可视化
- **`src/visualize_rpmnet.py`** - 原始可视化工具
  ```bash
  python src/visualize_rpmnet.py --dataset modelnet40 --resume checkpoints/model.pth
  ```

- **`src/arguments.py`** - 命令行参数定义
  ```bash
  # 定义所有命令行参数
  ```

#### 数学工具
- **`src/common/math/se3.py`** - SE(3)变换数学
- **`src/common/math/so3.py`** - SO(3)旋转数学
- **`src/common/math_torch/se3.py`** - PyTorch版本的SE(3)
- **`src/common/colors.py`** - 颜色定义
- **`src/common/misc.py`** - 杂项工具

#### 用户添加的文件
- **`src/choose_vis_idx.py`** - 选择可视化索引 **[ADDED]**
  ```bash
  # 帮助选择要可视化的数据索引
  ```

- **`src/sample_analysis.png`** - 样本分析图片 **[ADDED]**
  ```bash
  # 样本数据分析结果图片
  ```

---

## 🐌 雷达测试目录 (`snail_test/`)

### 核心脚本

#### 数据转换
- **`snail_test/convert_radar_single_frames.py`** - 雷达数据转换脚本 **[ADDED]**
  ```bash
  python convert_radar_single_frames.py
  # 将PCD雷达数据转换为RPM-Net可用的HDF5格式
  ```

#### 推理和测试
- **`snail_test/infer_single_pair.py`** - 单对点云推理脚本 **[ADDED]**
  ```bash
  python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5 --resume checkpoints/partial-trained.pth --src 0 --ref 3 --save_vis
  # 对单对雷达帧进行配准推理
  ```

- **`snail_test/process_sequence.py`** - 序列处理脚本 **[ADDED]**
  ```bash
  python process_sequence.py --h5 radar_full_sequence/radar_single_frames_test0.h5 --resume checkpoints/partial-trained.pth --start 0 --end 50
  # 处理雷达帧序列，生成轨迹
  ```

#### 合成测试
- **`snail_test/create_synthetic_test.py`** - 合成测试数据生成器 **[ADDED]**
  ```bash
  python create_synthetic_test.py --input eagleg7/enhanced/1696641884.835595373.pcd --output synthetic_test/ --rotation 30 0 15 --translation 2.0 1.0 0.5
  # 从单个PCD创建已知ground truth的测试数据
  ```

#### 可视化和分析
- **`snail_test/visualize_radar_registration.py`** - 雷达配准可视化工具 **[ADDED]**
  ```bash
  python visualize_radar_registration.py --results_dir radar_single_frames_original/ --mode progress --save progress.png
  # 可视化雷达数据的配准过程和结果
  ```

- **`snail_test/enhanced_visualization.py`** - 增强版可视化工具 **[ADDED]**
  ```bash
  python enhanced_visualization.py --results_dir radar_single_frames_original/ --save enhanced_progress.png
  # 提供6种不同视角的增强可视化
  ```

- **`snail_test/diagnose_registration.py`** - 配准诊断工具 **[ADDED]**
  ```bash
  python diagnose_registration.py --results_dir radar_single_frames_original/
  # 全面诊断配准问题并提供参数建议
  ```

#### 参数调优
- **`snail_test/parameter_tuning_guide.py`** - 参数调优工具 **[ADDED]**
  ```bash
  python parameter_tuning_guide.py --h5 radar_single_frames_original/radar_single_frames_test0.h5 --checkpoint checkpoints/partial-trained.pth --results_dir radar_single_frames_original/
  # 自动测试多种参数组合，找到最佳设置
  ```

#### 辅助工具
- **`snail_test/analyze_results.py`** - 结果分析工具 **[ADDED]**
  ```bash
  python analyze_results.py --results_dir radar_single_frames_original/
  # 分析配准结果的统计信息
  ```

- **`snail_test/validate_converted_data.py`** - 数据验证工具 **[ADDED]**
  ```bash
  python validate_converted_data.py --h5 radar_single_frames_original/radar_single_frames_test0.h5
  # 验证转换后的HDF5数据格式和内容
  ```

- **`snail_test/debug_helpers.py`** - 调试辅助工具 **[ADDED]**
  ```bash
  # 包含各种调试和辅助函数
  ```

- **`snail_test/test_setup.py`** - 测试环境设置 **[ADDED]**
  ```bash
  python test_setup.py
  # 检查测试环境和依赖
  ```

### 数据目录

#### 原始雷达数据
- **`snail_test/eagleg7/`** - Oculii雷达数据集 **[DATA]**
  ```
  eagleg7/
  ├── enhanced/          # 增强后的PCD文件
  ├── pcl/              # 原始PCL文件
  └── trk/              # 轨迹文件
  ```

#### 转换后的数据
- **`snail_test/radar_single_frames/`** - 固定点数转换结果 **[DATA]**
  ```
  radar_single_frames/
  ├── radar_single_frames_test0.h5    # HDF5数据文件
  ├── normals_log.txt                 # 法向量估计日志
  ├── pair_before.ply                 # 配准前点云
  ├── pair_after.ply                  # 配准后点云
  └── T_src0_ref1.txt                 # 变换矩阵
  ```

- **`snail_test/radar_single_frames_original/`** - 原始点数转换结果 **[DATA]**
  ```
  radar_single_frames_original/
  ├── radar_single_frames_test0.h5    # HDF5数据文件（变长点数）
  ├── frame_indices_mapping.txt       # 帧索引映射
  ├── pred_transforms.npy             # 所有迭代的变换矩阵
  ├── perm_matrices.pickle            # 匹配矩阵
  ├── data_dict.npy                   # 原始点云数据
  └── ... (其他结果文件)
  ```

#### 合成测试数据
- **`snail_test/synthetic_test/`** - 基本合成测试 **[DATA]**
  ```
  synthetic_test/
  ├── synthetic_test.h5               # 合成HDF5数据
  ├── ground_truth_transform.txt      # 真实变换
  ├── source.ply                      # 源点云
  ├── reference.ply                   # 目标点云
  └── ... (其他结果文件)
  ```

- **`snail_test/synthetic_partial/`** - 部分重叠合成测试 **[DATA]**
  ```
  synthetic_partial/
  ├── synthetic_test.h5               # 部分重叠的合成数据
  ├── ground_truth_transform.txt      # 真实变换
  └── ... (其他结果文件)
  ```

### 文档目录

#### 教程和指南
- **`snail_test/README.md`** - 项目说明 **[ADDED]**
  ```bash
  # 雷达测试项目的总体说明
  ```

- **`snail_test/QUICKSTART.md`** - 快速开始指南 **[ADDED]**
  ```bash
  # 快速上手指南，包含基本使用流程
  ```

- **`snail_test/QUICK_REFERENCE.md`** - 快速参考 **[ADDED]**
  ```bash
  # 常用命令和参数的快速参考
  ```

- **`snail_test/QUICK_FIX_GUIDE.md`** - 问题修复指南 **[ADDED]**
  ```bash
  # 常见问题的快速修复指南
  ```

#### 详细教程
- **`snail_test/VISUALIZATION_TUTORIAL.md`** - 可视化教程 **[ADDED]**
  ```bash
  # 详细的可视化工具使用教程
  ```

- **`snail_test/SEQUENCE_TESTING_TUTORIAL.md`** - 序列测试教程 **[ADDED]**
  ```bash
  # 序列处理和轨迹生成的详细教程
  ```

#### 总结文档
- **`snail_test/IMPLEMENTATION_SUMMARY.md`** - 实现总结 **[ADDED]**
  ```bash
  # 整个项目的实现总结
  ```

- **`snail_test/NEW_FEATURES_SUMMARY.md`** - 新功能总结 **[ADDED]**
  ```bash
  # 新增功能的详细总结
  ```

- **`snail_test/UPDATES_SUMMARY.md`** - 更新总结 **[ADDED]**
  ```bash
  # 所有更新的总结
  ```

### 其他文件
- **`snail_test/progress.png`** - 进度可视化图片 **[OUTPUT]**
  ```bash
  # 配准进度可视化结果图片
  ```

---

## 📊 数据文件说明

### HDF5 数据格式
```python
# 固定点数格式
data: (N_frames, 1024, 3)        # 点云坐标
normal: (N_frames, 1024, 3)      # 法向量
normal_is_random: (N_frames, 1024)  # 法向量是否随机
label: (N_frames,)               # 帧标签

# 变长点数格式
data_0, data_1, ...: (N_points_i, 3)  # 每帧的点云
normal_0, normal_1, ...: (N_points_i, 3)  # 每帧的法向量
normal_is_random_0, normal_is_random_1, ...: (N_points_i,)  # 每帧的随机掩码
```

### 结果文件格式
```python
# 变换矩阵
pred_transforms.npy: (1, n_iter, 3, 4)  # 所有迭代的变换矩阵

# 匹配矩阵
perm_matrices.pickle: [list of sparse matrices]  # 每迭代的匹配矩阵

# 原始数据
data_dict.npy: {
    'points_src': (N, 3),
    'points_ref': (N, 3),
    'normals_src': (N, 3),
    'normals_ref': (N, 3)
}
```

---

## 🚀 典型工作流

### 1. 数据转换
```bash
# 转换雷达数据为HDF5格式
python convert_radar_single_frames.py
```

### 2. 单对推理
```bash
# 运行单对配准
python infer_single_pair.py --h5 radar_single_frames_original/radar_single_frames_test0.h5 --resume checkpoints/partial-trained.pth --src 0 --ref 3 --save_vis
```

### 3. 可视化结果
```bash
# 查看配准进度
python visualize_radar_registration.py --results_dir radar_single_frames_original/ --mode progress --save progress.png

# 增强可视化
python enhanced_visualization.py --results_dir radar_single_frames_original/ --save enhanced.png
```

### 4. 诊断和调优
```bash
# 诊断问题
python diagnose_registration.py --results_dir radar_single_frames_original/

# 参数调优
python parameter_tuning_guide.py --h5 radar_single_frames_original/radar_single_frames_test0.h5 --checkpoint checkpoints/partial-trained.pth --results_dir radar_single_frames_original/
```

### 5. 合成测试
```bash
# 创建合成测试数据
python create_synthetic_test.py --input eagleg7/enhanced/1696641884.835595373.pcd --output synthetic_test/ --rotation 30 0 15 --translation 2.0 1.0 0.5

# 测试合成数据
python infer_single_pair.py --h5 synthetic_test/synthetic_test.h5 --resume checkpoints/partial-trained.pth --src 0 --ref 1 --save_vis
```

---

## 🏷️ 标签说明

- **[ADDED]** - 用户新增的文件
- **[DATA]** - 数据文件或目录
- **[OUTPUT]** - 输出结果文件
- 无标签 - RPM-Net 原始文件

---

## 📝 维护建议

1. **定期清理输出文件** - 删除不需要的 `.ply`, `.png` 等输出文件
2. **备份重要数据** - 保留 `eagleg7/` 原始数据和转换后的 HDF5 文件
3. **版本控制** - 只提交代码和文档，忽略数据和输出文件
4. **文档更新** - 添加新功能时及时更新相关文档

---

*最后更新: 2024年12月*
