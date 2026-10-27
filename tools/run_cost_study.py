"""Cost-function study for the article: all controllers x scenarios S1/S2.

Usage:  python tools/run_cost_study.py [--only NAME ...]
Writes results/cost_study/{results.json, traj/*.npz} and is post-processed by
tools/make_cost_report.py (figures + LaTeX tables for results_report/).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.append(os.path.dirname(__file__))
from study_setup import make_controller, all_names, SCENARIOS, S2_T_ON, S2_LEVELS, DT
from mpc_utils import PLANT, simulate_mpc, mse, mae, kvt

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'cost_study'))
G = PLANT["g"]


def theta_ss(Fw):
    """Upright equilibrium under a constant tip force: tan(theta) = -Fw / (m g)."""
    return -np.arctan(np.asarray(Fw) / (PLANT["m"] * G))


def stability(scen, X):
    """Automatic check; every run is still inspected visually in the report."""
    th, x = X[:, 0], X[:, 2]
    if not np.all(np.isfinite(X)) or np.max(np.abs(th)) > np.pi / 4 or np.max(np.abs(x)) > 5.0:
        return "unstable"
    k_ends = [len(X) - 1] if scen["key"] == "S1" else \
        [int(round(t / DT)) - 1 for t in S2_T_ON[1:]] + [len(X) - 1]  # last sample before each step
    th_targets = [0.0] if scen["key"] == "S1" else list(theta_ss(S2_LEVELS))
    for k, th_t in zip(k_ends, th_targets):
        if abs(X[k, 0] - th_t) > 5e-3 or abs(X[k, 1]) > 1e-2 or abs(X[k, 3]) > 1e-2:
            return "not settled"
    if scen["key"] == "S1" and abs(x[-1] - scen["x_ref"][2]) > 1e-2:
        return "not settled"
    return "stable"


def run_one(job):
    name, skey = job
    scen = SCENARIOS[skey]
    ctrl = make_controller(name)
    X, U, Fw, t_ctrl, _ = simulate_mpc(PLANT, ctrl, scen["x0"], scen["x_ref"], scen["T"], DT,
                                       wind=scen["wind"])
    # Indicators over k = 1..N_s (samples after the first control move), as in the article
    th, x = X[1:, 0], X[1:, 2]
    th_ref = np.full_like(th, scen["x_ref"][0])
    x_ref = np.full_like(x, scen["x_ref"][2])
    res = {
        "name": name, "scenario": skey,
        "mse_theta": mse(th, th_ref), "mae_theta": mae(th, th_ref),
        "mse_x": mse(x, x_ref), "mae_x": mae(x, x_ref),
        "kvt": kvt(U),
        "max_theta": float(np.max(np.abs(X[:, 0]))), "max_x": float(np.max(np.abs(X[:, 2]))),
        "max_u": float(np.max(np.abs(U))),
        "t_ctrl_ms": 1e3 * t_ctrl / len(U),
        "solver_calls": int(getattr(ctrl, "n_calls", 0)),
        "solver_fails": int(getattr(ctrl, "n_fail", 0)),
        "status": stability(scen, X),
    }
    np.savez(os.path.join(OUT_DIR, "traj", f"{skey}_{name}.npz"), X=X, U=U, Fw=Fw)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these controllers (merged into results.json)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT_DIR, "traj"), exist_ok=True)
    names = args.only or all_names()
    jobs = [(n, s) for s in SCENARIOS for n in names]

    res_file = os.path.join(OUT_DIR, "results.json")
    results = {}
    if args.only and os.path.exists(res_file):
        results = json.load(open(res_file, encoding="utf-8"))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_one, jobs):
            results[f"{r['scenario']}|{r['name']}"] = r
            print(f"{r['scenario']} {r['name']:12s} {r['status']:12s} MSEth={r['mse_theta']:.3e} "
                  f"MSEx={r['mse_x']:.3e} KVT={r['kvt']:.3f} fails={r['solver_fails']}/{r['solver_calls']} "
                  f"t={r['t_ctrl_ms']:.1f}ms", flush=True)

    json.dump(results, open(res_file, "w", encoding="utf-8"), indent=2)
    print("saved", res_file)


if __name__ == "__main__":
    main()
