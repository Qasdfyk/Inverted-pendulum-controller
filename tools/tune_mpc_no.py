import os, sys, time, math, random, argparse
from pathlib import Path
import numpy as np, yaml

# --- Headless mode for speed ---
os.environ["CARTPOLE_HEADLESS"] = "1"

# --- Ensure project root import ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sim import run

CTRL_DIR = ROOT / "config" / "controllers"

def _write_yaml(path: Path, block: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(block, f, sort_keys=False)

def _rm(path: Path):
    try: path.unlink()
    except FileNotFoundError: pass

def _objective(temp_variant_name: str):
    res = run(controller="mpc_no", disturbance=False, animation=False,
              step_type="position_step", archive_results=False,
              config_variant=temp_variant_name)
    mse_th = res.metrics["mse_theta"]
    mse_x  = res.metrics["mse_x"]
    u_rms  = float(np.sqrt(np.mean(res.U**2)))
    J = mse_th + 0.3*mse_x + 0.01*(u_rms**2)
    return J, res

def _trial(cfg_block: dict, trial_idx: int):
    tmp_name = f"_tmp_mpc_no_{trial_idx}_{int(time.time()*1000)}"
    tmp_path = CTRL_DIR / f"{tmp_name}.yaml"
    _write_yaml(tmp_path, cfg_block)
    try:
        return _objective(tmp_name)
    finally:
        _rm(tmp_path)

def sample_params():
    return {
        "mpc_no": {
            "dt": 0.02,
            "H": random.choice([2,3,4]),
            "u_min": -10.0,
            "u_max": 10.0,
            "Q_theta": random.uniform(5, 50),
            "Q_x": random.uniform(0.5, 10),
            "R": random.uniform(0.001, 0.2),
        }
    }

def run_search(n_trials=40, seed=42, verbose=True):
    random.seed(seed)
    best = {"J": math.inf, "cfg": None}
    for k in range(n_trials):
        cfg_block = sample_params()
        try: J, _ = _trial(cfg_block, k)
        except Exception as e:
            if verbose: print(f"[{k:03d}] failed: {e}")
            continue
        if verbose:
            print(f"[{k:03d}] J={J:.5f} H={cfg_block['mpc_no']['H']} "
                  f"Qθ={cfg_block['mpc_no']['Q_theta']:.1f} Qx={cfg_block['mpc_no']['Q_x']:.1f} R={cfg_block['mpc_no']['R']:.3f}")
        if J < best["J"]:
            best = {"J": J, "cfg": cfg_block}
    return best

def main():
    ap = argparse.ArgumentParser(description="Tune MPC-NO controller")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="mpc_no_tuned")
    args = ap.parse_args()

    print(f"=== Tuning MPC-NO (headless) trials={args.trials} seed={args.seed} ===")
    best = run_search(n_trials=args.trials, seed=args.seed)
    if best["cfg"] is None:
        print("No successful trials."); return

    out_path = CTRL_DIR / f"{args.out}.yaml"
    _write_yaml(out_path, best["cfg"])
    print(f"\nBest J={best['J']:.6f} -> saved to {out_path.name}")
    print(f"Run with: python main.py --controller mpc_no --config-variant {args.out} --disturbance off --animation off --step position_step")

if __name__ == "__main__":
    main()
