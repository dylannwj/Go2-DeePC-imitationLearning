"""Minimal public surface for the released DeePC implementation."""

from .geometry import cumulative_matrix, difference_matrix, wrap_angle
from .hankel import HankelData, load_hankel
from .qp import DeePCSolution, DeePCSolverError, QPConfig, solve_qp

__all__ = [
    "cumulative_matrix",
    "difference_matrix",
    "wrap_angle",
    "HankelData",
    "load_hankel",
    "DeePCSolution",
    "DeePCSolverError",
    "QPConfig",
    "solve_qp",
]
