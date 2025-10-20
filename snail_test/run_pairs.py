# run_pairs.py
import csv, subprocess, sys
from pathlib import Path

# ==== 按需改这里 ====
INFER_SCRIPT = Path("infer_from_pcd.py")  # 单对脚本
VIS_SCRIPT   = Path("visualize_radar_registration.py")  # 可视化脚本
RESUME       = Path(r"D:/AA_projects_in_nus/nus/deep_sparse_radar_odometry/code/checkpoints/partial-trained.pth")
TRAJ_CSV     = Path("ref_trajs/fwd_bwd_loc/20231007/data4/ref_tls_T_oculii.csv")
OUT_DIR_BASE = Path("outputs/enhanced_batch2")
# OUT_DIR_BASE = Path("outputs/pcl_batch3")
PCD_DIR      = Path("eagleg7/enhanced") 
# PCD_DIR      = Path("eagleg7/pcl") 

# 公共参数（想用 auto_radius 就把 "--radius","0.3"两项去掉，换成 "--auto_radius"）
COMMON_ARGS = ["--num_iter","5","--radius","0.3","--neighbors","64","--save_vis"]
PAIRS_CSV   = Path(r"enhanced_pairs.csv")           # CSV 路径
# PAIRS_CSV   = Path(r"pairs.csv")           # CSV 路径
PYEXE       = sys.executable                         # 用当前python
DRY_RUN     = False                                   # 只打印不执行：True/False
# =====================

def main():
    OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    cfg = (f"INFER_SCRIPT={INFER_SCRIPT}\nVIS_SCRIPT={VIS_SCRIPT}\nRESUME={RESUME}\nTRAJ_CSV={TRAJ_CSV}\n"
       f"OUT_DIR_BASE={OUT_DIR_BASE}\nCOMMON_ARGS={' '.join(COMMON_ARGS)}\nPAIRS_CSV={PAIRS_CSV}\nPYEXE={PYEXE}\nDRY_RUN={DRY_RUN}\n")
    (OUT_DIR_BASE / "run_config.txt").write_text(cfg, encoding="utf-8")

    with PAIRS_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert {"src_pcd","ref_pcd"} <= set(reader.fieldnames or []), "CSV需含表头: src_pcd,ref_pcd[,out_subdir]"
        for row in reader:
            src = (PCD_DIR / row["src_pcd"]).expanduser()
            ref = (PCD_DIR / row["ref_pcd"]).expanduser()
            out_sub = (row.get("out_subdir") or "").strip()
            out_dir = OUT_DIR_BASE / (out_sub if out_sub else f"pair_{src.stem}_{ref.stem}")
            out_dir.mkdir(parents=True, exist_ok=True)

            infer_cmd = [
                PYEXE, str(INFER_SCRIPT),
                "--src_pcd", str(src),
                "--ref_pcd", str(ref),
                "--resume", str(RESUME),
                "--traj_csv", str(TRAJ_CSV),
                "--out_dir", str(out_dir),
                *COMMON_ARGS,
            ]

            print(">>", " ".join(infer_cmd))
            rc = 0
            if not DRY_RUN:
                rc = subprocess.run(infer_cmd, check=False).returncode

            # 2) 可视化命令（仅当推理成功时执行；想强制可视化，把条件去掉）
            if rc == 0:
                vis_cmd = [
                    PYEXE, str(VIS_SCRIPT),
                    "--results_dir", str(out_dir),
                    "--save", str(out_dir / "progress.png"),
                ]
                print(">>", " ".join(vis_cmd))
                if not DRY_RUN:
                    subprocess.run(vis_cmd, check=False)
            else:
                print(f"[WARN] 推理返回码 {rc}，跳过可视化：{out_dir}")

if __name__ == "__main__":
    main()
