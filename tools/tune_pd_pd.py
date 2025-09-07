import os, sys, time, math, random, argparse
from pathlib import Path
import numpy as np
import yaml

# --- Headless for speed (no plots during tuning) ---
os.environ["CARTPOLE_HEADLESS"] = "1"

# --- Ensure project root is importable ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports (now that ROOT is on sys.path)
from env.sim import run

CTRL_DIR = ROOT / "config" / "controllers"

def _write_yaml(path: Path, block: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(block, f, sort_keys=False)

def _rm(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass

def _objective(temp_variant_name: str):
    """Run one sim using a variant name (must exist in config/controllers)."""
    res = run(controller="pd_pd", disturbance=False, animation=False,
              step_type="position_step", archive_results=False,
              config_variant=temp_variant_name)
    mse_th = res.metrics["mse_theta"]
    mse_x  = res.metrics["mse_x"]
    u_rms  = float(np.sqrt(np.mean(res.U**2)))
    # Scalar objective: tweak to preference
    J = mse_th + 0.2*mse_x + 0.001*(u_rms**2)
    return J, res

def _trial(cfg_block: dict, trial_idx: int):
    """
    Create a TEMP variant yaml inside config/controllers, evaluate it, then delete it.
    Returns (J, res). Ensures cleanup even on errors.
    """
    tmp_name = f"_tmp_pd_pd_{trial_idx}_{int(time.time()*1000)}"
    tmp_path = CTRL_DIR / f"{tmp_name}.yaml"
    _write_yaml(tmp_path, cfg_block)
    try:
        return _objective(tmp_name)
    finally:
        _rm(tmp_path)

def sample_params():
    """Randomized but conservative ranges that tend to stabilize without saturation."""
    return {
        "cart_pid": {
            "Kp":  random.uniform(0.10, 0.30),
            "Ki":  random.choice([0.0, random.uniform(0.0, 0.02)]),
            "Kd":  random.uniform(0.04, 0.12),
        },
        "theta_ref_limit":       random.uniform(0.08, 0.15),  # keep small tilts
        "theta_ref_rate":        random.uniform(0.3, 0.8),    # rad/s
        "theta_ref_filter_tau":  random.uniform(0.08, 0.22),  # s
        "pend_pid": {
            "Kp":  random.uniform(10.0, 18.0),
            "Ki":  0.0,
            "Kd":  random.uniform(3.0, 7.0),
        }
    }

def run_search(n_trials=60, seed=23341, verbose=True):
    random.seed(seed)
    best = {"J": math.inf, "cfg": None}
    for k in range(n_trials):
        cfg_block = sample_params()
        try:
            J, _ = _trial(cfg_block, k)
        except Exception as e:
            if verbose:
                print(f"[{k:03d}] failed: {e}")
            continue
        if verbose:
            print(f"[{k:03d}] J={J:.5f}  "
                  f"cart(Kp={cfg_block['cart_pid']['Kp']:.3f},Ki={cfg_block['cart_pid']['Ki']:.3f},Kd={cfg_block['cart_pid']['Kd']:.3f})  "
                  f"pend(Kp={cfg_block['pend_pid']['Kp']:.2f},Kd={cfg_block['pend_pid']['Kd']:.2f})  "
                  f"θlim={cfg_block['theta_ref_limit']:.3f} rate={cfg_block['theta_ref_rate']:.2f} tau={cfg_block['theta_ref_filter_tau']:.2f}")
        if J < best["J"]:
            best = {"J": J, "cfg": cfg_block}
    return best

def main():
    ap = argparse.ArgumentParser(description="Tune PD–PD controller (only write best config).")
    ap.add_argument("--trials", type=int, default=60, help="number of random trials")
    ap.add_argument("--seed", type=int, default=23341, help="random seed")
    ap.add_argument("--out", type=str, default="pd_pd_tuned", help="output config variant name (without .yaml)")
    args = ap.parse_args()

    print(f"=== Tuning PD–PD (headless) trials={args.trials} seed={args.seed} ===")
    best = run_search(n_trials=args.trials, seed=args.seed, verbose=True)

    if best["cfg"] is None:
        print("No successful trials. Try increasing --trials or relaxing parameter ranges.")
        return

    out_name = args.out
    out_path = CTRL_DIR / f"{out_name}.yaml"
    _write_yaml(out_path, best["cfg"])
    print(f"\nBest J={best['J']:.6f} -> saved to {out_path.name}")
    print("Run it with:")
    print(f"  python main.py --controller pd_pd --config-variant {out_name} --disturbance off --animation off --step position_step --archive off")

if __name__ == "__main__":
    main()
