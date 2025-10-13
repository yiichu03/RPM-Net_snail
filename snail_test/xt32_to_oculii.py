import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

in_csv = r"snail_test\ref_trajs\fwd_bwd_loc\20231007\data4\ref_tls_T_xt32.csv"
out_csv = r"snail_test\ref_trajs\fwd_bwd_loc\20231007\data4\ref_tls_T_oculii.csv"

xt32_T_oculii = np.array([
       [-0.0198,   0.9994,    0.0274,   -0.0000],
       [-0.9998,   -0.0200,   0.0093,   -0.0700],
       [0.0098,    -0.0272,   0.9996,   -0.1150],
       [0,         0,         0,    1.0000],
], dtype=float)

def T_from_pq(p_xyz, q_xyzw):
    '''
    由位置+四元数得到齐次变换矩阵'''
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = p_xyz
    return T

def pq_from_T(T):
    '''
    由齐次变换矩阵得到位置+四元数'''
    p_xyz = T[:3, 3]
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return p_xyz, q_xyzw


df = pd.read_csv(in_csv, header=None, comment='#')
df.columns = [
    "time",
    "M_p_L_x", "M_p_L_y", "M_p_L_z",
    "M_q_L_x", "M_q_L_y", "M_q_L_z", "M_q_L_w",
    "W_v_L_x", "W_v_L_y", "W_v_L_z",
]

rows = []
for _, r in df.iterrows():
    W_p_x = np.array([r.M_p_L_x, r.M_p_L_y, r.M_p_L_z], dtype=float)
    W_q_x = np.array([r.M_q_L_x, r.M_q_L_y, r.M_q_L_z, r.M_q_L_w], dtype=float)
    W_T_x = T_from_pq(W_p_x, W_q_x)

    W_T_o = W_T_x @ xt32_T_oculii
    W_p_o, W_q_o = pq_from_T(W_T_o)

    rows.append([r.time, W_p_o[0], W_p_o[1], W_p_o[2], W_q_o[0], W_q_o[1], W_q_o[2], W_q_o[3]])

out_cols = ["time", "M_p_O_x", "M_p_O_y", "M_p_O_z", "M_q_O_x", "M_q_O_y", "M_q_O_z", "M_q_O_w"]
pd.DataFrame(rows, columns=out_cols).to_csv(out_csv, index=False)
print("Saved:", out_csv)