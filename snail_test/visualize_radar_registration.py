"""
Visualize RPM-Net registration progress (top view, XY only).
仅可视化顶视图(XY)下，源点云随每次迭代对齐到参考点云的过程。
Expected files under --results_dir:
  - pred_transforms.npy : shape (B, n_iter, 3, 4) or (n_iter, 3, 4) or (3, 4)
  - data_dict.npy       : dict with keys ['points_src', 'points_ref', ...]
Usage:
  python visualize_radar_registration.py \
      --results_dir path/to/results_dir \
      --save progress.png

1.读结果
   - Load pred_transforms.npy → normalize to shape (n_iter, 3, 4). 若为(B, n_iter, 3, 4)只取第一个样本；若为(3,4)则扩展为(1,3,4)。
   - Load data_dict.npy → get points_src, points_ref (N×3).
2.工具
   - apply_transform(P, T): 应用 SE(3) 变换 T（R|t）到点集 P。
   - rotation_deg(T): 由 trace(R) 计算合成旋转角（度）。
   - translation_norm(T): 计算平移向量范数（米）。

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: str):
    """Load transforms and point clouds."""
    # 1) transforms
    tr_path = os.path.join(results_dir, "pred_transforms.npy")
    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"pred_transforms.npy not found in: {results_dir}")
    tr = np.load(tr_path)  # (B, n_iter, 3, 4) or (n_iter, 3, 4) or (3, 4)

    # Normalize shapes to (n_iter, 3, 4)
    if tr.ndim == 4:                 # (B, n_iter, 3, 4)
        if tr.shape[0] != 1:
            print(f"[warn] B={tr.shape[0]} > 1; only visualizing the first sample.")
        tr = tr[0]                   # -> (n_iter, 3, 4)
    elif tr.ndim == 3:               # (n_iter, 3, 4) or (3,4)
        if tr.shape == (3, 4):
            tr = tr[None, ...]       # -> (1, 3, 4)
    else:
        raise ValueError(f"Unexpected pred_transforms shape: {tr.shape}")

    # 2) points
    dd_path = os.path.join(results_dir, "data_dict.npy")
    if not os.path.exists(dd_path):
        raise FileNotFoundError(f"data_dict.npy not found in: {results_dir}")
    data_dict = np.load(dd_path, allow_pickle=True).item()
    pts_src = np.asarray(data_dict["points_src"], dtype=np.float32)
    pts_ref = np.asarray(data_dict["points_ref"], dtype=np.float32)

    return tr, pts_src, pts_ref


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply SE3 transform T (3x4 or 4x4) to Nx3 points."""
    # We only need the top-left 3x3 and top-right 3x1
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def rotation_deg(T: np.ndarray) -> float:
    """Compute rotation angle (deg) from transform."""
    R = T[:3, :3]
    tr = np.trace(R)
    ang = np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    return float(ang * 180.0 / np.pi)


def translation_norm(T: np.ndarray) -> float:
    """Compute translation magnitude (meters) from transform."""
    t = T[:3, 3]
    return float(np.linalg.norm(t))


def plot_progress(transforms: np.ndarray,
                  pts_src: np.ndarray,
                  pts_ref: np.ndarray,
                  save_path: str = None,
                  cols: int = 3):
    """
    Plot initial (before) + each iteration result in a grid.
    - transforms: (n_iter, 3, 4)
    - subsample to at most max_points for speed/clarity
    """
    n_iter = transforms.shape[0]


    n_panels = n_iter + 1
    cols = max(1, cols)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    fig.suptitle("RPM-Net Registration Progress (Top view: XY)", fontsize=16)

    # Panel 0: initial
    ax0 = axes[0]
    ax0.scatter(pts_src[:, 0], pts_src[:, 1], s=3, c="red", alpha=0.6, label="Source")
    ax0.scatter(pts_ref[:, 0], pts_ref[:, 1], s=3, c="blue", alpha=0.6, label="Reference")
    ax0.set_title("Initial (Before)")
    ax0.legend()
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.3)
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")

    # Panels 1..n_iter: after each iteration
    for i in range(n_iter):
        ax = axes[i + 1]
        T = transforms[i]
        src_aligned = apply_transform(pts_src, T)

        ax.scatter(src_aligned[:, 0], src_aligned[:, 1], s=3, c="green", alpha=0.6, label="Aligned Source")
        ax.scatter(pts_ref[:, 0], pts_ref[:, 1], s=3, c="blue", alpha=0.6, label="Reference")
        ax.set_title(f"Iter {i + 1}\nRot: {rotation_deg(T):.2f}°, Trans: {translation_norm(T):.3f} m")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # Hide unused axes
    for j in range(n_panels, rows * cols):
        axes[j].remove()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved figure to:", save_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser("Progress visualization (top view)")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Folder with pred_transforms.npy and data_dict.npy")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save the figure (e.g., progress.png)")
    parser.add_argument("--cols", type=int, default=3, help="可视化：子图网格一行放几个面板")
    args = parser.parse_args()

    tr, pts_src, pts_ref = load_results(args.results_dir)
    print(f"[load] transforms: {tr.shape}, src: {pts_src.shape}, ref: {pts_ref.shape}")
    plot_progress(tr, pts_src, pts_ref, save_path=args.save, cols=args.cols)


if __name__ == "__main__":
    main()
