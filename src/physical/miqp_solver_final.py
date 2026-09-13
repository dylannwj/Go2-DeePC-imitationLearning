#!/usr/bin/env python3
"""
Fast free-g, strict-input, deadzone-aware DeePC MIQP preview solver for Go2.

Purpose:
- Closer to textbook/regularized DeePC than the simplex variant.
- g is free, but regularized with ||g||^2.
- Future input consistency is strict by default: Uf g == uf.
- Discrete/deadzone command levels are enforced with binary variables.
- Past input/output matching may use slack for noisy real data.

Offline preview only. Do not send to robot until JSON/CSV are sane.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    print("ERROR: gurobipy import failed:", e)
    sys.exit(1)


def load_hankel(path: str):
    z = np.load(path, allow_pickle=True)
    for k in ["Up", "Uf", "Yp", "Yf"]:
        if k not in z.files:
            raise KeyError(f"Missing key {k}; found {z.files}")
    return z, z["Up"].astype(float), z["Uf"].astype(float), z["Yp"].astype(float), z["Yf"].astype(float)


def infer_dims(Up, Uf, Yp, Yf):
    if Up.shape[0] % 3 or Uf.shape[0] % 3:
        raise ValueError(f"Expected input rows multiple of 3. Up={Up.shape}, Uf={Uf.shape}")
    if Yp.shape[0] % 3 or Yf.shape[0] % 3:
        raise ValueError(f"Expected output rows multiple of 3. Yp={Yp.shape}, Yf={Yf.shape}")
    return Up.shape[0] // 3, Uf.shape[0] // 3


def select_columns(Up, Uf, Yp, Yf, N_use, max_cols, target):
    cols = Uf.shape[1]
    if max_cols is None or max_cols <= 0 or max_cols >= cols:
        idx = np.arange(cols)
        return Up, Uf, Yp, Yf, idx

    UfN = Uf[: 3 * N_use, :]
    YfN = Yf[: 3 * N_use, :]

    vx_seq = UfN[0::3, :]
    vy_seq = UfN[1::3, :]
    yaw_seq = UfN[2::3, :]
    fwd = YfN[0::3, :].sum(axis=0)
    left = YfN[1::3, :].sum(axis=0)
    yaw = YfN[2::3, :].sum(axis=0)

    target_fwd, target_left, target_yaw = target

    score = (
        3000.0 * (fwd - target_fwd) ** 2
        + 5000.0 * (left - target_left) ** 2
        + 8000.0 * (yaw - target_yaw) ** 2
    )

    if target_fwd > 0.002:
        score -= 250.0 * np.sum(vx_seq >= 0.10 - 1e-9, axis=0)
        score += 20000.0 * (fwd <= 0.0)
        score += 1000.0 * np.sum(np.abs(yaw_seq), axis=0)

    if target_fwd < -0.002:
        score -= 500.0 * np.sum(vx_seq <= -0.10 + 1e-9, axis=0)
        score += 20000.0 * (fwd >= 0.0)
        score += 1000.0 * np.sum(np.abs(yaw_seq), axis=0)

    if abs(target_left) > 0.002:
        sign_l = 1.0 if target_left > 0 else -1.0
        score -= 500.0 * np.sum(sign_l * vy_seq >= 0.10 - 1e-9, axis=0)
        score += 20000.0 * (sign_l * left <= 0.0)
        score += 500.0 * np.sum(np.abs(yaw_seq), axis=0)

    if abs(target_yaw) > 0.02:
        sign = 1.0 if target_yaw > 0 else -1.0
        score -= 250.0 * np.sum(sign * yaw_seq >= 0.20 - 1e-9, axis=0)
        score += 2000.0 * np.sum(np.abs(vx_seq), axis=0)
        score += 2000.0 * np.sum(np.abs(vy_seq), axis=0)

    idx = np.argsort(score)[:max_cols]
    return Up[:, idx], Uf[:, idx], Yp[:, idx], Yf[:, idx], idx


def qsum_square(v, n=None):
    if n is None:
        n = len(v)
    return gp.quicksum(v[i] * v[i] for i in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hankel", required=True)
    ap.add_argument("--u-ini-file", default=None)
    ap.add_argument("--y-ini-file", default=None)
    ap.add_argument("--force-first-forward", action="store_true")
    ap.add_argument("--force-first-vx-sign", choices=["none","pos","neg"], default="none")
    ap.add_argument("--force-first-yaw-sign", choices=["none","pos","neg"], default="none")
    ap.add_argument("--force-first-vy-sign", choices=["none","pos","neg"], default="none")
    ap.add_argument("--target-forward", type=float, default=0.0)
    ap.add_argument("--target-left", type=float, default=0.0)
    ap.add_argument("--target-yaw", type=float, default=0.0)
    ap.add_argument("--use-n", type=int, default=20)
    ap.add_argument("--max-cols", type=int, default=80)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--mip-gap", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out-csv", default="miqp_freeg_strict_preview.csv")
    ap.add_argument("--out-json", default="miqp_freeg_strict_preview.json")

    ap.add_argument("--w-terminal-forward", type=float, default=1_000_000.0)
    ap.add_argument("--w-terminal-left", type=float, default=300_000.0)
    ap.add_argument("--w-terminal-yaw", type=float, default=1_000_000.0)
    ap.add_argument("--w-stage-forward", type=float, default=5_000.0)
    ap.add_argument("--w-stage-left", type=float, default=5_000.0)
    ap.add_argument("--w-stage-yaw", type=float, default=50_000.0)
    ap.add_argument("--w-effort-vx", type=float, default=0.0)
    ap.add_argument("--w-effort-vy", type=float, default=0.0)
    ap.add_argument("--w-effort-yaw", type=float, default=200_000.0)
    ap.add_argument("--w-smooth-vx", type=float, default=0.0)
    ap.add_argument("--w-smooth-vy", type=float, default=0.0)
    ap.add_argument("--w-smooth-yaw", type=float, default=50_000.0)
    ap.add_argument("--w-sigma-up", type=float, default=10_000.0)
    ap.add_argument("--w-sigma-yp", type=float, default=10_000.0)
    ap.add_argument("--w-g", type=float, default=100.0)

    ap.add_argument("--force-yaw-zero", action="store_true")
    ap.add_argument("--force-forward-nonzero", action="store_true")
    ap.add_argument("--min-forward-steps", type=int, default=0)
    ap.add_argument("--force-yaw-nonzero", action="store_true")
    ap.add_argument("--simplex-g", action="store_true", help="Constrain g >= 0 and sum(g)=1 to prevent cancellation.")
    ap.add_argument("--allow-uf-slack", action="store_true", help="Allow Uf g to differ from discrete uf. Default is strict/no-cheat.")
    ap.add_argument("--w-sigma-uf", type=float, default=1_000_000.0)
    args = ap.parse_args()

    z, Up, Uf, Yp, Yf = load_hankel(args.hankel)
    Tini, N_full = infer_dims(Up, Uf, Yp, Yf)
    if args.use_n < 1 or args.use_n > N_full:
        raise ValueError(f"--use-n must be 1..{N_full}, got {args.use_n}")
    N = args.use_n

    target = np.array([args.target_forward, args.target_left, args.target_yaw], dtype=float)
    Up, Uf, Yp, Yf, idx = select_columns(Up, Uf, Yp, Yf, N, args.max_cols, target)
    UfN = Uf[: 3 * N, :]
    YfN = Yf[: 3 * N, :]
    cols = Up.shape[1]

    if args.u_ini_file:
        u_ini = np.load(args.u_ini_file).astype(float).reshape(-1)
    else:
        u_ini = np.zeros(3 * Tini)

    if args.y_ini_file:
        y_ini = np.load(args.y_ini_file).astype(float).reshape(-1)
    else:
        y_ini = np.zeros(3 * Tini)

    if u_ini.shape[0] != 3 * Tini:
        raise ValueError(f"u_ini wrong length: {u_ini.shape[0]} expected {3*Tini}")
    if y_ini.shape[0] != 3 * Tini:
        raise ValueError(f"y_ini wrong length: {y_ini.shape[0]} expected {3*Tini}")

    m = gp.Model("fast_freeg_strict_deadzone_miqp_deepc")
    m.Params.TimeLimit = args.time_limit
    m.Params.MIPGap = args.mip_gap
    m.Params.Threads = args.threads
    m.Params.DualReductions = 0

    # Textbook-style DeePC: free Hankel mixing weights, regularized in the objective.
    # No future-input cheating is allowed because Uf g == uf by default below.
    g = m.addMVar(cols, lb=0.0 if args.simplex_g else -GRB.INFINITY, name="g")
    if args.simplex_g:
        m.addConstr(gp.quicksum(g[i] for i in range(cols)) == 1.0, name="simplex_g_sum")

    sigma_up = m.addMVar(3 * Tini, lb=-GRB.INFINITY, name="sigma_up")
    sigma_yp = m.addMVar(3 * Tini, lb=-GRB.INFINITY, name="sigma_yp")
    uf = m.addMVar(3 * N, lb=-GRB.INFINITY, name="uf")
    yf = m.addMVar(3 * N, lb=-GRB.INFINITY, name="yf")

    m.addConstr(Up @ g + sigma_up == u_ini, name="past_input_soft")
    m.addConstr(Yp @ g + sigma_yp == y_ini, name="past_output_soft")

    if args.allow_uf_slack:
        sigma_uf = m.addMVar(3 * N, lb=-GRB.INFINITY, name="sigma_uf")
        m.addConstr(UfN @ g - uf + sigma_uf == np.zeros(3 * N), name="future_input_soft")
    else:
        sigma_uf = None
        m.addConstr(UfN @ g == uf, name="future_input_strict")

    m.addConstr(YfN @ g == yf, name="future_output_prediction")

    vx_levels = [-0.10, 0.0, 0.10, 0.15]
    neg_vx_idx = 0
    pos_vx_idxs = [2, 3]
    vy_levels = [-0.10, 0.0, 0.10]
    neg_vy_idx = 0
    pos_vy_idx = 2
    yaw_levels = [-0.30, -0.20, 0.0, 0.20, 0.30]
    zero_yaw_idx = 2
    pos_yaw_idxs = [3, 4]
    neg_yaw_idxs = [0, 1]

    z_vx = m.addVars(N, len(vx_levels), vtype=GRB.BINARY, name="z_vx")
    z_vy = m.addVars(N, len(vy_levels), vtype=GRB.BINARY, name="z_vy")
    z_yaw = m.addVars(N, len(yaw_levels), vtype=GRB.BINARY, name="z_yaw")

    for t in range(N):
        m.addConstr(gp.quicksum(z_vx[t, j] for j in range(len(vx_levels))) == 1, name=f"one_vx_{t}")
        m.addConstr(gp.quicksum(z_vy[t, j] for j in range(len(vy_levels))) == 1, name=f"one_vy_{t}")
        m.addConstr(gp.quicksum(z_yaw[t, j] for j in range(len(yaw_levels))) == 1, name=f"one_yaw_{t}")
        m.addConstr(uf[3*t] == gp.quicksum(vx_levels[j] * z_vx[t, j] for j in range(len(vx_levels))), name=f"vx_level_{t}")
        m.addConstr(uf[3*t+1] == gp.quicksum(vy_levels[j] * z_vy[t, j] for j in range(len(vy_levels))), name=f"vy_level_{t}")
        m.addConstr(uf[3*t+2] == gp.quicksum(yaw_levels[j] * z_yaw[t, j] for j in range(len(yaw_levels))), name=f"yaw_level_{t}")
        if args.force_yaw_zero:
            m.addConstr(z_yaw[t, zero_yaw_idx] == 1, name=f"force_yaw_zero_{t}")

    if args.force_forward_nonzero:
        m.addConstr(gp.quicksum(z_vx[t, 1] + z_vx[t, 2] for t in range(N)) >= 1, name="force_some_forward")

    if args.min_forward_steps > 0:
        m.addConstr(gp.quicksum(z_vx[t, 1] + z_vx[t, 2] for t in range(N)) >= args.min_forward_steps, name="min_forward_steps")

    if args.force_yaw_nonzero:
        if args.target_yaw >= 0:
            m.addConstr(gp.quicksum(z_yaw[t, j] for t in range(N) for j in pos_yaw_idxs) >= 1, name="force_pos_yaw")
        else:
            m.addConstr(gp.quicksum(z_yaw[t, j] for t in range(N) for j in neg_yaw_idxs) >= 1, name="force_neg_yaw")

    if args.force_first_forward:
        m.addConstr(gp.quicksum(z_vx[0, j] for j in pos_vx_idxs) >= 1, name="force_first_forward_0")

    if args.force_first_vx_sign == "pos":
        m.addConstr(gp.quicksum(z_vx[0, j] for j in pos_vx_idxs) >= 1, name="force_first_pos_vx_0")
    elif args.force_first_vx_sign == "neg":
        m.addConstr(z_vx[0, neg_vx_idx] == 1, name="force_first_neg_vx_0")

    if args.force_first_yaw_sign == "pos":
        m.addConstr(gp.quicksum(z_yaw[0, j] for j in pos_yaw_idxs) >= 1, name="force_first_pos_yaw_0")
    elif args.force_first_yaw_sign == "neg":
        m.addConstr(gp.quicksum(z_yaw[0, j] for j in neg_yaw_idxs) >= 1, name="force_first_neg_yaw_0")

    if args.force_first_vy_sign == "pos":
        m.addConstr(z_vy[0, pos_vy_idx] == 1, name="force_first_pos_vy_0")
    elif args.force_first_vy_sign == "neg":
        m.addConstr(z_vy[0, neg_vy_idx] == 1, name="force_first_neg_vy_0")

    obj = gp.QuadExpr()

    fwd_terminal = gp.quicksum(yf[3*t] for t in range(N))
    left_terminal = gp.quicksum(yf[3*t+1] for t in range(N))
    yaw_terminal = gp.quicksum(yf[3*t+2] for t in range(N))

    obj += args.w_terminal_forward * (fwd_terminal - args.target_forward) * (fwd_terminal - args.target_forward)
    obj += args.w_terminal_left * (left_terminal - args.target_left) * (left_terminal - args.target_left)
    obj += args.w_terminal_yaw * (yaw_terminal - args.target_yaw) * (yaw_terminal - args.target_yaw)

    for t in range(N):
        frac = float(t + 1) / float(N)
        cf = gp.quicksum(yf[3*k] for k in range(t+1))
        cl = gp.quicksum(yf[3*k+1] for k in range(t+1))
        cy = gp.quicksum(yf[3*k+2] for k in range(t+1))
        obj += args.w_stage_forward * (cf - frac * args.target_forward) * (cf - frac * args.target_forward)
        obj += args.w_stage_left * (cl - frac * args.target_left) * (cl - frac * args.target_left)
        obj += args.w_stage_yaw * (cy - frac * args.target_yaw) * (cy - frac * args.target_yaw)
        obj += args.w_effort_vx * uf[3*t] * uf[3*t]
        obj += args.w_effort_vy * uf[3*t+1] * uf[3*t+1]
        obj += args.w_effort_yaw * uf[3*t+2] * uf[3*t+2]
        if t > 0:
            dv = uf[3*t] - uf[3*(t-1)]
            dv_y = uf[3*t+1] - uf[3*(t-1)+1]
            dw = uf[3*t+2] - uf[3*(t-1)+2]
            obj += args.w_smooth_vx * dv * dv
            obj += args.w_smooth_vy * dv_y * dv_y
            obj += args.w_smooth_yaw * dw * dw

    obj += args.w_sigma_up * qsum_square(sigma_up, 3*Tini)
    obj += args.w_sigma_yp * qsum_square(sigma_yp, 3*Tini)
    if sigma_uf is not None:
        obj += args.w_sigma_uf * qsum_square(sigma_uf, 3*N)
    obj += args.w_g * qsum_square(g, cols)

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status = status_map.get(m.Status, str(m.Status))
    print("Solver status:", status)

    if m.SolCount < 1:
        result = {
            "status": status,
            "solution_count": int(m.SolCount),
            "hankel": args.hankel,
            "Tini": Tini,
            "N_used": N,
            "N_full": N_full,
            "columns_used": int(cols),
            "strict_uf": not args.allow_uf_slack,
            "simplex_g": False,
            "message": "No solution available; no CSV written.",
        }
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print("Wrote", args.out_json)
        return

    uf_val = np.array(uf.X).reshape(-1)
    yf_val = np.array(yf.X).reshape(-1)
    g_val = np.array(g.X).reshape(-1)
    sup = np.array(sigma_up.X).reshape(-1)
    syp = np.array(sigma_yp.X).reshape(-1)
    suf = np.zeros(3*N) if sigma_uf is None else np.array(sigma_uf.X).reshape(-1)

    vx = uf_val[0::3]
    vy = uf_val[1::3]
    wz = uf_val[2::3]
    fwd_step = yf_val[0::3]
    left_step = yf_val[1::3]
    yaw_step = yf_val[2::3]
    fwd_cum = np.cumsum(fwd_step)
    left_cum = np.cumsum(left_step)
    yaw_cum = np.cumsum(yaw_step)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "vx_cmd", "vy_cmd", "yaw_rate_cmd",
            "pred_forward_step_m", "pred_left_step_m", "pred_yaw_step_rad",
            "pred_forward_cum_m", "pred_left_cum_m", "pred_yaw_cum_rad"
        ])
        writer.writeheader()
        for t in range(N):
            writer.writerow({
                "step": t,
                "vx_cmd": float(vx[t]),
                "vy_cmd": float(vy[t]),
                    "yaw_rate_cmd": float(wz[t]),
                "pred_forward_step_m": float(fwd_step[t]),
                "pred_left_step_m": float(left_step[t]),
                "pred_yaw_step_rad": float(yaw_step[t]),
                "pred_forward_cum_m": float(fwd_cum[t]),
                "pred_left_cum_m": float(left_cum[t]),
                "pred_yaw_cum_rad": float(yaw_cum[t]),
            })

    result = {
        "status": status,
        "solution_count": int(m.SolCount),
        "objective": float(m.ObjVal),
        "runtime_sec": float(m.Runtime),
        "mip_gap": float(m.MIPGap),
        "hankel": args.hankel,
        "Tini": Tini,
        "N_used": N,
        "N_full": N_full,
        "columns_used": int(cols),
        "max_cols_requested": int(args.max_cols),
        "simplex_g": False,
        "strict_uf": not args.allow_uf_slack,
        "target": {
            "forward_m": float(args.target_forward),
            "left_m": float(args.target_left),
            "yaw_rad": float(args.target_yaw),
        },
        "terminal_prediction": {
            "forward_m": float(fwd_cum[-1]),
            "left_m": float(left_cum[-1]),
            "yaw_rad": float(yaw_cum[-1]),
        },
        "terminal_error": {
            "forward_m": float(fwd_cum[-1] - args.target_forward),
            "left_m": float(left_cum[-1] - args.target_left),
            "yaw_rad": float(yaw_cum[-1] - args.target_yaw),
        },
        "vx_unique": sorted(float(x) for x in np.unique(np.round(vx, 6))),
        "yaw_unique": sorted(float(x) for x in np.unique(np.round(wz, 6))),
        "sigma_norms": {
            "up": float(np.linalg.norm(sup)),
            "yp": float(np.linalg.norm(syp)),
            "uf": float(np.linalg.norm(suf)),
        },
        "g_sum": float(np.sum(g_val)),
        "g_min": float(np.min(g_val)),
        "g_max": float(np.max(g_val)),
        "g_l2_norm": float(np.linalg.norm(g_val)),
        "g_nonzero_abs_gt_1e_minus_6": int(np.sum(np.abs(g_val) > 1e-6)),
        "selected_column_indices_preview": [int(x) for x in idx[:20]],
        "out_csv": args.out_csv,
        "out_json": args.out_json,
        "note": "FAST FREE-G STRICT 3INPUT DOG2: g is free/regularized, and Uf g == commanded uf unless --allow-uf-slack is used. Offline preview only.",
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))

    print("Wrote", args.out_csv)
    print("Wrote", args.out_json)
    print("terminal prediction:", result["terminal_prediction"])
    print("vx_unique:", result["vx_unique"])
    print("yaw_unique:", result["yaw_unique"])
    print("g_norm:", result["g_l2_norm"], "g_min:", result["g_min"], "g_max:", result["g_max"])


if __name__ == "__main__":
    main()
