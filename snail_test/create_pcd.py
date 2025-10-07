"""
Create a transformed copy of a PCD with given SE(3) (no downsampling).

Usage:
  python create_pcd.py \
      --input eagleg7/enhanced/1696641884.835595373.pcd \
      --output out_dir/ \
      --rotation 30 0 15 \
      --translation 2.0 1.0 0.5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def euler_xyz_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation matrix from XYZ Euler angles (degrees). R = Rz * Ry * Rx."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float64)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx



def main():
    parser = argparse.ArgumentParser("Create transformed PCD (no downsampling)")
    parser.add_argument("--input", type=str, required=True, help="Input .pcd")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--rotation", nargs=3, type=float, required=True,
                        help="Euler XYZ angles in degrees: rx ry rz")
    parser.add_argument("--translation", nargs=3, type=float, required=True,
                        help="Translation in meters: tx ty tz")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"PCD not found: {in_path}")

    # 1) Load points (no downsampling)
    pcd = o3d.io.read_point_cloud(str(in_path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("Empty PCD!")

    # 2) Build SE(3)
    rx, ry, rz = args.rotation
    tx, ty, tz = args.translation
    R = euler_xyz_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float64)
    T34 = np.zeros((3, 4), dtype=np.float64)
    T34[:, :3] = R
    T34[:, 3] = t

    # 3) Transform points
    pts_out = pts @ R.T + t

    # 4) Compose output names
    stem = in_path.stem
    suffix = f"_R_{rx}_{ry}_{rz}_T_{tx}_{ty}_{tz}"
    pcd_out = out_dir / f"{stem}{suffix}.pcd"
    T_readable = out_dir / f"T_{stem}{suffix}.txt"

    # 5) Save transformed PCD (same number of points)
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_out))
    o3d.io.write_point_cloud(str(pcd_out), pcd2)
    print(f"[save] PCD  -> {pcd_out}")

    # 7) (Readable) Also save R & t with labels for humans
    with open(T_readable, "w", encoding="utf-8") as f:
        f.write("R (3x3):\n")
        np.savetxt(f, R, fmt="%.9f")
        f.write("\n")
        f.write("t (3,):\n")
        np.savetxt(f, t[None, :], fmt="%.9f")
    print(f"[save] RT readable -> {T_readable}")

    # Console summary
    rot_trace = np.trace(R)
    rot_rad = np.arccos(np.clip((rot_trace - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(rot_rad * 180.0 / np.pi)
    t_norm = float(np.linalg.norm(t))
    print("\n=== Ground Truth SE(3) ===")
    print("R (3x3):\n", R)
    print("t (m): ", t)
    print(f"rotation ≈ {rot_deg:.3f} deg, translation = {t_norm:.3f} m")


if __name__ == "__main__":
    main()
