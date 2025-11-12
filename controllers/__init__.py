
from .pid_lqr import build as build_pid_lqr
from .pd_pd import build as build_pd_pd


REGISTRY = {
    "pid_lqr": build_pid_lqr,
    "pd_pd": build_pd_pd,
}
