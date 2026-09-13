"""Constrained quadratic DeePC solver.

The formulation follows the maintained three-input receding-horizon
implementation: cumulative output tracking, input effort, past consistency,
input smoothness, and a regularized Hankel coefficient vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import cumulative_matrix, difference_matrix
from .hankel import HankelData


class DeePCSolverError(RuntimeError):
    """Raised when the configured QP cannot produce a valid solution."""


@dataclass(frozen=True)
class QPConfig:
    """Convex DeePC objective and safety envelopes."""

    tracking_weights: tuple[float, ...] = ()
    input_weights: tuple[float, ...] = ()
    past_input_weight: float = 50.0
    past_output_weight: float = 200.0
    delta_input_weight: float = 1.0
    g_weight: float = 1e-4
    input_lower: tuple[float, ...] | None = None
    input_upper: tuple[float, ...] | None = None
    max_delta_input: tuple[float, ...] | None = None
    solver: str = "OSQP"
    solver_eps_abs: float = 1e-6
    solver_eps_rel: float = 1e-6
    solver_max_iter: int = 100_000

    def validate(self, u_dim: int, y_dim: int) -> None:
        for name, values in (
            ("tracking_weights", self.tracking_weights),
            ("input_weights", self.input_weights),
        ):
            if len(values) not in (y_dim if name == "tracking_weights" else u_dim, 0):
                raise ValueError(f"{name} has the wrong dimension")
            if values and (not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0)):
                raise ValueError(f"{name} must contain positive finite values")
        for name, value in (
            ("past_input_weight", self.past_input_weight),
            ("past_output_weight", self.past_output_weight),
            ("delta_input_weight", self.delta_input_weight),
            ("g_weight", self.g_weight),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.input_lower is not None and len(self.input_lower) != u_dim:
            raise ValueError("input_lower has the wrong dimension")
        if self.input_upper is not None and len(self.input_upper) != u_dim:
            raise ValueError("input_upper has the wrong dimension")
        if (
            self.input_lower is not None
            and self.input_upper is not None
            and np.any(np.asarray(self.input_lower) > np.asarray(self.input_upper))
        ):
            raise ValueError("input_lower must not exceed input_upper")
        if self.max_delta_input is not None and (
            len(self.max_delta_input) != u_dim or np.any(np.asarray(self.max_delta_input) < 0)
        ):
            raise ValueError("max_delta_input must be nonnegative with input dimension")

    def resolved_tracking_weights(self, y_dim: int) -> np.ndarray:
        """Return configured output weights, or dimension-safe defaults."""

        if self.tracking_weights:
            return np.asarray(self.tracking_weights, dtype=float)
        default = (8.0, 12.0, 8.0)
        return np.asarray(
            tuple(
                default[index] if index < len(default) else default[-1] for index in range(y_dim)
            ),
            dtype=float,
        )

    def resolved_input_weights(self, u_dim: int) -> np.ndarray:
        """Return configured input weights, or a dimension-safe default."""

        if self.input_weights:
            return np.asarray(self.input_weights, dtype=float)
        return np.full(u_dim, 0.15, dtype=float)


@dataclass(frozen=True)
class DeePCSolution:
    """Solver result in time-major array form."""

    command: np.ndarray
    planned_input: np.ndarray
    predicted_output: np.ndarray
    g: np.ndarray
    status: str
    objective: float
    solver_name: str


def _validate_history(
    hankel: HankelData, u_ini: np.ndarray, y_ini: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u_ini, dtype=float)
    y = np.asarray(y_ini, dtype=float)
    expected_u = (hankel.T_ini, hankel.u_dim)
    expected_y = (hankel.T_ini, hankel.y_dim)
    if u.shape != expected_u or y.shape != expected_y:
        raise ValueError(f"history shapes must be {expected_u} and {expected_y}")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(y)):
        raise ValueError("history contains NaN or Inf")
    return u, y


def solve_qp(
    hankel: HankelData,
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    target_cumulative_output: np.ndarray,
    config: QPConfig | None = None,
    *,
    previous_plan: np.ndarray | None = None,
) -> DeePCSolution:
    """Solve one constrained DeePC preview and return its first command."""

    config = config or QPConfig()
    config.validate(hankel.u_dim, hankel.y_dim)
    u_history, y_history = _validate_history(hankel, u_ini, y_ini)
    target = np.asarray(target_cumulative_output, dtype=float)
    if target.shape != (hankel.N, hankel.y_dim):
        raise ValueError(f"target_cumulative_output must have shape {(hankel.N, hankel.y_dim)}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains NaN or Inf")
    if previous_plan is not None:
        previous = np.asarray(previous_plan, dtype=float)
        if previous.shape != (hankel.N, hankel.u_dim):
            raise ValueError("previous_plan has the wrong shape")
        if not np.all(np.isfinite(previous)):
            raise ValueError("previous_plan contains NaN or Inf")
    else:
        previous = None

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise DeePCSolverError("CVXPY is required for the QP backend") from exc

    columns = hankel.columns
    coefficient = cp.Variable(columns, name="g")
    future_input = hankel.Uf @ coefficient
    future_output = hankel.Yf @ coefficient
    cumulative_output = cumulative_matrix(hankel.N, hankel.y_dim) @ future_output
    track_weight = np.sqrt(np.tile(config.resolved_tracking_weights(hankel.y_dim), hankel.N))
    input_weight = np.sqrt(np.tile(config.resolved_input_weights(hankel.u_dim), hankel.N))
    objective_terms = [
        cp.sum_squares(cp.multiply(track_weight, cumulative_output - target.reshape(-1))),
        cp.sum_squares(cp.multiply(input_weight, future_input)),
        config.past_input_weight * cp.sum_squares(hankel.Up @ coefficient - u_history.reshape(-1)),
        config.past_output_weight * cp.sum_squares(hankel.Yp @ coefficient - y_history.reshape(-1)),
        config.g_weight * cp.sum_squares(coefficient),
    ]
    if config.delta_input_weight and hankel.N > 1:
        delta = difference_matrix(hankel.N, hankel.u_dim) @ future_input
        objective_terms.append(config.delta_input_weight * cp.sum_squares(delta))
    if previous is not None:
        objective_terms.append(cp.sum_squares(future_input - previous.reshape(-1)))

    constraints = []
    if config.input_lower is not None:
        constraints.append(future_input >= np.tile(config.input_lower, hankel.N))
    if config.input_upper is not None:
        constraints.append(future_input <= np.tile(config.input_upper, hankel.N))
    if config.max_delta_input is not None and hankel.N > 1:
        delta = difference_matrix(hankel.N, hankel.u_dim) @ future_input
        limits = np.tile(config.max_delta_input, hankel.N - 1)
        constraints.extend((delta >= -limits, delta <= limits))

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    solver_name = config.solver
    try:
        problem.solve(
            solver=getattr(cp, solver_name),
            eps_abs=config.solver_eps_abs,
            eps_rel=config.solver_eps_rel,
            max_iter=config.solver_max_iter,
            warm_start=True,
            verbose=False,
        )
    except (cp.error.SolverError, AttributeError) as exc:
        fallback = cp.CLARABEL if hasattr(cp, "CLARABEL") else None
        if fallback is None:
            raise DeePCSolverError(f"configured solver {solver_name!r} is unavailable") from exc
        solver_name = "CLARABEL"
        problem.solve(solver=fallback, warm_start=True, verbose=False)
    if coefficient.value is None or future_input.value is None or future_output.value is None:
        raise DeePCSolverError(f"QP did not produce a solution: {problem.status}")
    planned_input = np.asarray(future_input.value, dtype=float).reshape(hankel.N, hankel.u_dim)
    predicted_output = np.asarray(future_output.value, dtype=float).reshape(hankel.N, hankel.y_dim)
    values = (planned_input, predicted_output, np.asarray(coefficient.value, dtype=float))
    if not all(np.all(np.isfinite(value)) for value in values):
        raise DeePCSolverError("QP returned a non-finite solution")
    return DeePCSolution(
        command=planned_input[0].copy(),
        planned_input=planned_input,
        predicted_output=predicted_output,
        g=values[2],
        status=str(problem.status),
        objective=float(problem.value),
        solver_name=solver_name,
    )
