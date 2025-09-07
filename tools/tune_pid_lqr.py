import os, sys, time, math, random, argparse
from pathlib import Path
import numpy as np, yaml

# --- Headless mode (no plots) ---
os.environ["CARTPOLE_HEADLESS"] = "1"

# --- Ensure project root is importable ---
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
    res = run(controller="pid_lqr", disturbance=False, animation=False,
              step_type="position_step", archive_results=False,
              config_variant=temp_variant_name)
    mse_th = res.metrics["mse_theta"]
    mse_x  = res.metrics["mse_x"]
    u_rms  = float(np.sqrt(np.mean(res.U**2)))
    J = mse_th + 0.2*mse_x + 0.001*(u_rms**2)
    return J, res

def _trial(cfg_block: dict, trial_idx: int):
    tmp_name = f"_tmp_pid_lqr_{trial_idx}_{int(time.time()*1000)}"
    tmp_path = CTRL_DIR / f"{tmp_name}.yaml"
    _write_yaml(tmp_path, cfg_block)
    try:
        return _objective(tmp_name)
    finally:
        _rm(tmp_path)

def sample_params():
    return {
        "cart_pid": {
            "Kp":  random.uniform(0.5, 2.0),
            "Ki":  random.uniform(0.0, 0.5),
            "Kd":  random.uniform(0.0, 0.5),
        },
        "lqr": {
            "Q": [random.uniform(10,200),
                  random.uniform(1,50),
                  random.uniform(50,400),
                  random.uniform(1,50)],
            "R": random.uniform(0.05, 2.0),
        }
    }

def run_search(n_trials=50, seed=123, verbose=True):
    random.seed(seed)
    best = {"J": math.inf, "cfg": None}
    for k in range(n_trials):
        cfg_block = sample_params()
        try: J, _ = _trial(cfg_block, k)
        except Exception as e:
            if verbose: print(f"[{k:03d}] failed: {e}")
            continue
        if verbose:
            print(f"[{k:03d}] J={J:.5f} cart(Kp={cfg_block['cart_pid']['Kp']:.2f},Ki={cfg_block['cart_pid']['Ki']:.2f},Kd={cfg_block['cart_pid']['Kd']:.2f}) "
                  f"LQR(Q={cfg_block['lqr']['Q']}, R={cfg_block['lqr']['R']:.3f})")
        if J < best["J"]:
            best = {"J": J, "cfg": cfg_block}
    return best

def main():
    ap = argparse.ArgumentParser(description="Tune PID+LQR controller")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=23341)
    ap.add_argument("--out", type=str, default="pid_lqr_tuned")
    args = ap.parse_args()

    print(f"=== Tuning PID+LQR (headless) trials={args.trials} seed={args.seed} ===")
    best = run_search(n_trials=args.trials, seed=args.seed)
    if best["cfg"] is None:
        print("No successful trials."); return

    out_path = CTRL_DIR / f"{args.out}.yaml"
    _write_yaml(out_path, best["cfg"])
    print(f"\nBest J={best['J']:.6f} -> saved to {out_path.name}")
    print(f"Run with: python main.py --controller pid_lqr --config-variant {args.out} --disturbance off --animation off --step position_step --archive off")

if __name__ == "__main__":
    main()
