# Radar Point Cloud Registration Tools

以下是主要脚本的功能介绍和使用方法。

## 数据准备
数据下载网址：https://snail-radar.github.io/

我目前只试了20231007的data4，用了其中的eagleg7文件夹；以及ref_trajs，用了对应的轨迹文件（xt32的，然后用外参 https://github.com/snail-radar/dataset_tools/tree/main/matlab 转到oculii）。

- 点云数据（PCD文件）: 我放在了 `snail_test/eagleg7/pcl/` ,`snail_test/eagleg7/enhanced/`
- 轨迹数据（CSV文件）: 我放在了`snail_test/ref_trajs`

## 主要工具

### 1. RPMNet模型推理 (infer_from_pcd.py)

从两个PCD文件中推断相对变换，使用RPMNet模型进行点云配准。

```bash
python infer_from_pcd.py --resume path/to/model.pth --src_pcd path/to/source.pcd --ref_pcd path/to/reference.pcd --traj_csv path/to/ref_tls_T_oculii.csv --out_dir output_directory --save_vis
```

参数说明:
- `--resume`: RPMNet模型检查点路径
- `--src_pcd`: 源点云PCD文件路径
- `--ref_pcd`: 参考点云PCD文件路径
- `--traj_csv`: 轨迹CSV文件路径
- `--out_dir`: 输出目录
- `--save_vis`: 保存可视化结果
- `--auto_radius`: 自动估计特征半径
- `--radius`: 特征半径（当不使用自动估计时）
- `--neighbors`: 每个点的邻居数量
- `--num_iter`: RPMNet注册迭代次数

> auto_radius是之前gpt建议的，但是我看他和直接设置radius的效果是差不多，所以直接按照RPMNet原本的设置一个radius值（或者就直接用他给的默认值）就可以


### 2. 一次运行多对的脚本 (run_pairs.py)

批量处理点云配对，根据CSV文件中定义的配对关系进行处理。

```bash
python run_pairs.py --pairs_csv pairs.csv --model_path path/to/model.pth --pcd_dir data/pcl --out_dir results
```

参数说明:
- `--pairs_csv`: 包含点云配对信息的CSV文件
- `--model_path`: RPMNet模型路径
- `--pcd_dir`: 点云数据目录
- `--out_dir`: 结果输出目录

### 3. 可视化雷达配准 (visualize_radar_registration.py)

可视化点云配准结果，展示源点云、目标点云和配准后的点云。

```bash
python visualize_radar_registration.py --src_pcd path/to/source.pcd --ref_pcd path/to/reference.pcd --transform_txt path/to/transform.txt
```

参数说明:
- `--src_pcd`: 源点云PCD文件路径
- `--ref_pcd`: 参考点云PCD文件路径
- `--transform_txt`: 变换矩阵文本文件路径

### 4. 合并点云与轨迹 (merge_pcds_with_traj.py)

读取PCD序列和轨迹数据，将点云合并到同一坐标系中。

```bash
python merge_pcds_with_traj.py --pcd_dir data/pcl --traj_csv data/ref_trajs/ref_tls_T_oculii.csv --out_pcd merged_full.pcd --traj_points_out traj_points.pcd
```

参数说明:
- `--pcd_dir`: PCD序列目录
- `--traj_csv`: 轨迹CSV文件路径
- `--out_pcd`: 合并点云输出文件名
- `--traj_points_out`: 轨迹位置点输出文件名
- `--time_tolerance`: 允许的最近邻时间差(秒)

## 其他
- 所有PCD文件名格式为时间戳（如 `1696641886.183674812.pcd`）
- 轨迹CSV文件包含时间、位置和姿态（四元数）信息