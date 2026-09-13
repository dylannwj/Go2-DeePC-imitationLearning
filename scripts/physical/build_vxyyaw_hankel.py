#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("dog2_vxyyaw_increment_dataset_ACTIVE_ONLY_2026_06_19.csv")
OUT = Path("dog2_vxyyaw_hankel_Tini10_N20_2026_06_19.npz")
META = Path("dog2_vxyyaw_hankel_column_metadata_2026_06_19.csv")

TINI = 10
N = 20
INPUT_COLS = ["safe_vx", "safe_vy", "safe_yaw_rate"]
OUTPUT_COLS = ["forward_step", "left_step", "yaw_step"]

df = pd.read_csv(IN)
df = df.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

U_cols = []
Y_cols = []
meta = []

for source, g in df.groupby("source_file", sort=False):
    g = g.reset_index(drop=True)

    U = g[INPUT_COLS].to_numpy(float)
    Y = g[OUTPUT_COLS].to_numpy(float)

    L = TINI + N
    for start in range(0, len(g) - L + 1):
        u_win = U[start:start+L]
        y_win = Y[start:start+L]

        # Reject bad DeePC columns:
        # future says zero command, but output predicts meaningful movement.
        # This caused MIQP to choose vx=vy=wz=0 while predicting lateral motion.
        u_future = u_win[TINI:]
        y_future = y_win[TINI:]

        future_cmd_energy = float(np.sum(np.abs(u_future)))
        future_motion = float(np.linalg.norm(np.sum(y_future, axis=0)))

        if future_cmd_energy < 1e-9 and future_motion > 0.01:
            continue

        # Also reject nearly-zero command windows with large predicted movement.
        if future_cmd_energy < 0.20 and future_motion > 0.05:
            continue

        # time-major flatten: [u0..., u1..., ...]
        U_cols.append(u_win.reshape(-1))
        Y_cols.append(y_win.reshape(-1))

        meta.append({
            "source_file": source,
            "start_index": start,
            "timestamp_start": float(g.loc[start, "timestamp"]),
            "timestamp_end": float(g.loc[start+L-1, "timestamp"]),
            "vx_abs_max": float(np.max(np.abs(u_win[:,0]))),
            "vy_abs_max": float(np.max(np.abs(u_win[:,1]))),
            "yaw_abs_max": float(np.max(np.abs(u_win[:,2]))),
            "terminal_forward": float(np.sum(y_win[TINI:,0])),
            "terminal_left": float(np.sum(y_win[TINI:,1])),
            "terminal_yaw": float(np.sum(y_win[TINI:,2])),
        })

U_all = np.array(U_cols).T
Y_all = np.array(Y_cols).T

m = len(INPUT_COLS)
p = len(OUTPUT_COLS)

Up = U_all[:TINI*m, :]
Uf = U_all[TINI*m:, :]
Yp = Y_all[:TINI*p, :]
Yf = Y_all[TINI*p:, :]

np.savez(
    OUT,
    Up=Up, Uf=Uf, Yp=Yp, Yf=Yf,
    H_u_all=U_all,
    H_y_all=Y_all,
    Tini=np.array([TINI]),
    N=np.array([N]),
    input_dim=np.array([m]),
    output_dim=np.array([p]),
    input_cols=np.array(INPUT_COLS),
    output_cols=np.array(OUTPUT_COLS),
)

pd.DataFrame(meta).to_csv(META, index=False)

print("wrote", OUT)
print("columns", Up.shape[1])
print("Up", Up.shape, "Uf", Uf.shape, "Yp", Yp.shape, "Yf", Yf.shape)
print("input_dim", m, "output_dim", p)
print("metadata", META)
