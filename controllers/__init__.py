
from .pid_lqr import build as build_pid_lqr
from .pd_pd import build as build_pd_pd
from .lmpc import build as build_lmpc
from .mpc_no import build as build_mpc_no
from .mpc_altcost import build as build_mpc_altcost
from .ts_fuzzy import build as build_ts_fuzzy

REGISTRY = {
    "pid_lqr": build_pid_lqr,
    "pd_pd": build_pd_pd,
    "lmpc": build_lmpc,
    "mpc_no": build_mpc_no,
    "mpc_altcost": build_mpc_altcost,
    "ts_fuzzy": build_ts_fuzzy,
}
