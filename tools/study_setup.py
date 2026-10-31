"""Controllers and scenarios shared by the cost-function study (article experiments).

All controller parameters are the tuned values from the thesis (latex/tex/sterowanie.tex),
except the sign of Ki in PID-LQR; LQR and LQR-I were tuned with tuning/tune_lqr.py using the
same objective.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from pd_pd import PDPDController
from pd_lqr import PDLQRController
from lqr import LQRController
from lqr_i import LQRIController
from mpc_J2 import MPCControllerJ2
from lmpc import LinearMPCController
from fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant
from mpc_cost import MPCCostController, RHO
from mpc_utils import PLANT, SIM, StepDisturbance

DT = SIM["dt"]
U_SAT = SIM["u_sat"]
Q_NMPC = np.diag([158.39, 40.80, 43.41, 19.71])
R_NMPC = 0.08592

# Tuned parameters (reported in the PDF / article tables)
PARAMS = {
    "PID":       {"ang_pid": {"Kp": -40.0, "Ki": -1.0, "Kd": -8.0},
                  "cart_pid": {"Kp": -1.0, "Ki": -0.1, "Kd": -3.0}},
    "LQR":       {"Q": [437.398, 56.197, 378.065, 77.873], "R": 8.3412},
    "LQR-I":     {"Q": [148.887, 51.76, 326.26, 41.484, 0.06], "R": 5.9162},
    # Ki < 0, same sign as Kp: the thesis value Ki = +0.1 gives a closed-loop pole at 1.001
    "PID-LQR":   {"pid_gains": {"Kp": -7.0, "Ki": -0.1, "Kd": -3.0},
                  "lqr_gains": {"Q": [200.0, 3.0, 35.0, 40.0], "R": 1.0}},
    "Fuzzy-LQR": {"du_max": 800.0},
    "LMPC":      {"N": 12, "Nu": 4, "Q": [15.0, 1.0, 15.0, 1.0], "R": 0.1},
    "MPC-NO-alt": {"N": 12, "Nu": 4, "Q": list(np.diag(Q_NMPC)), "R": 0.001, "R_abs": 1.0},
    "NMPC":      {"N": 12, "Nu": 4, "Q": list(np.diag(Q_NMPC)), "R": R_NMPC, "c": 1.0},
}

BASELINE = ["PID", "LQR", "LQR-I", "PID-LQR", "Fuzzy-LQR", "LMPC", "MPC-NO-alt"]
NMPC_VARIANTS = list(RHO.keys())  # L2 first, then the 15 M-estimators


def make_controller(name: str):
    p = PARAMS
    if name == "PID":
        return PDPDController(PLANT, DT, **p["PID"])
    if name == "LQR":
        return LQRController(PLANT, DT, p["LQR"])
    if name == "LQR-I":
        return LQRIController(PLANT, DT, p["LQR-I"])
    if name == "PID-LQR":
        return PDLQRController(PLANT, DT, **p["PID-LQR"])
    if name == "Fuzzy-LQR":
        return TSFuzzyController(PLANT, starter_ts_params16(U_SAT), lqr_from_plant(PLANT), DT,
                                 du_max=p["Fuzzy-LQR"]["du_max"])
    if name == "LMPC":
        q = p["LMPC"]
        return LinearMPCController(PLANT, DT, q["N"], q["Nu"], -U_SAT, U_SAT, np.diag(q["Q"]), q["R"])
    if name == "MPC-NO-alt":
        q = p["MPC-NO-alt"]
        return MPCControllerJ2(PLANT, DT, q["N"], q["Nu"], -U_SAT, U_SAT, np.diag(q["Q"]), q["R"], q["R_abs"])
    if name.startswith("NMPC-"):
        q = p["NMPC"]
        return MPCCostController(PLANT, DT, q["N"], q["Nu"], -U_SAT, U_SAT, np.diag(q["Q"]), q["R"],
                                 rho=name[len("NMPC-"):], c=q["c"])
    raise KeyError(name)


def all_names():
    return BASELINE + [f"NMPC-{k}" for k in NMPC_VARIANTS]


# Scenario S1: nominal (as in the thesis)
S1 = {"key": "S1", "x0": SIM["x0"], "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),
      "T": SIM["T"], "wind": None}

# Scenario S2: disturbance steps from the upright equilibrium (supervisor's request)
S2_T_ON = [2.0, 22.0, 42.0, 62.0]
S2_LEVELS = [0.05, -0.10, 0.08, -0.04]  # [N]; theta_ss = -atan(Fw/(m g)) cannot be zero
S2 = {"key": "S2", "x0": np.zeros(4), "x_ref": np.zeros(4),
      "T": 82.0, "wind": StepDisturbance(S2_T_ON, S2_LEVELS)}

SCENARIOS = {"S1": S1, "S2": S2}
