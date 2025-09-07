
import argparse, os, sys
# ensure local imports work when running as: python main.py
sys.path.append(os.path.dirname(__file__))

from env.sim import run

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", required=True,
                   choices=["pid_lqr","pd_pd","lmpc","mpc_no","mpc_altcost","ts_fuzzy"])
    p.add_argument("--disturbance", choices=["on","off"], default="off")
    p.add_argument("--animation", choices=["on","off"], default="off")
    p.add_argument("--step", dest="step_type",
                   choices=["position_step","angle_step","track_sine","impulse"],
                   default="position_step")
    p.add_argument("--archive", choices=["on","off"], default="off")
    p.add_argument("--config-variant", default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        controller=args.controller,
        disturbance=(args.disturbance=="on"),
        animation=(args.animation=="on"),
        step_type=args.step_type,
        archive_results=(args.archive=="on"),
        config_variant=args.config_variant
    )
