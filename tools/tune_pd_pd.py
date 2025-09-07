import os, time, json, math, random
from pathlib import Path
import numpy as np
import yaml

# Make sim headless for speed
os.environ["CARTPOLE_HEADLESS"] = "1"

# Ensure project root in sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from utils.cfg import load_configs
from env.sim import run


ROOT = Path(__file__).resolve().parents[1]
CTRL_DIR = ROOT / "config" / "controllers"

def write_variant_yaml(name: str, block: dict):
    path = CTRL_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(block, f, sort_keys=False)
    return path

def objective(trial_cfg_name: str, t_end=6.0):
    # Temporarily shorten horizon for speed by overriding env.yaml at runtime would be ideal,
    # but we’ll just run and trust the default; you can also lower t_end in config/env.yaml while tuning.
    res = run(controller="pd_pd", disturbance=False, animation=False,
              step_type="position_step", archive_results=False,
              config_variant=trial_cfg_name)
    mse_th = res.metrics["mse_theta"]
    mse_x  = res.metrics["mse_x"]
    u_rms  = float(np.sqrt(np.mean(res.U**2)))
    # Multi-objective scalarization (tweak weights to taste)
    J = mse_th + 0.2*mse_x + 0.001*(u_rms**2)
    return J, res

def sample_params():
    # Reasonable sampling ranges (pure PD by default; tiny outer Ki optional)
    return {
        "cart_pid": {
            "Kp":  random.uniform(0.08, 0.35),
            "Ki":  random.choice([0.0, random.uniform(0.0, 0.03)]),
            "Kd":  random.uniform(0.02, 0.15),
        },
        "theta_ref_limit": random.uniform(0.08, 0.18),
        "theta_ref_rate":  random.uniform(0.3, 0.9),
        "theta_ref_filter_tau": random.uniform(0.08, 0.25),
        "pend_pid": {
            "Kp":  random.uniform(8.0, 18.0),
            "Ki":  0.0,
            "Kd":  random.uniform(2.0, 7.0),
        }
    }

def run_search(n_trials=50, seed=123):
    random.seed(seed)

    best = {"J": math.inf, "cfg": None, "name": None}
    for k in range(n_trials):
        cfg_block = sample_params()
        trial_name = f"pd_pd_trial_{k}"
        write_variant_yaml(trial_name, cfg_block)
        try:
            J, _ = objective(trial_name)
        except Exception as e:
            print(f"[{trial_name}] failed: {e}")
            continue
        print(f"[{k:03d}] J={J:.4f}  cart(Kp={cfg_block['cart_pid']['Kp']:.3f},Kd={cfg_block['cart_pid']['Kd']:.3f})  "
              f"pend(Kp={cfg_block['pend_pid']['Kp']:.1f},Kd={cfg_block['pend_pid']['Kd']:.1f})  "
              f"theta_ref_limit={cfg_block['theta_ref_limit']:.3f}")
        if J < best["J"]:
            best = {"J": J, "cfg": cfg_block, "name": trial_name}

    return best

def main():
    print("=== Tuning PD–PD (headless) ===")
    best = run_search(n_trials=60, seed=23341)
    if best["cfg"] is None:
        print("No successful trials.")
        return

    tuned_name = "pd_pd_tuned"
    write_variant_yaml(tuned_name, best["cfg"])
    print(f"\nBest J={best['J']:.5f} -> saved to {tuned_name}.yaml")
    print("Run it with:")
    print(f"  python main.py --controller pd_pd --config-variant {tuned_name} --disturbance off --animation off --step position_step --archive off")

if __name__ == "__main__":
    main()
