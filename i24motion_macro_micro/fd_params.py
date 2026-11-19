"""The one trusted LWR calibration: the triangular fundamental diagram and the
lane-change rate every first-order run uses.
"""
from __future__ import annotations

from typing import Dict

# Best LWR-triangular calibration (the set the demos, SUMO runs and RL were
# built around). Units: v_f and w in m/s, rho_j in veh/m per lane, lambda_lc
# in 1/s.
CANONICAL_FD_PARAMS: Dict[str, float] = {
    "v_f": 25.02031797294094,
    "w": 5.728460762748048,
    "rho_j": 0.07719493079089293,
}
CANONICAL_LAMBDA_LC: float = 0.13384902076274044


def canonical_params() -> Dict[str, float]:
    """FD parameters plus lambda_lc, in the dict shape the scripts used inline."""
    return {**CANONICAL_FD_PARAMS, "lambda_lc": CANONICAL_LAMBDA_LC}


def canonical_triangular_fd():
    from simulation import TriangularFD

    return TriangularFD(v_f=CANONICAL_FD_PARAMS["v_f"], w=CANONICAL_FD_PARAMS["w"],
                        rho_j=CANONICAL_FD_PARAMS["rho_j"])