"""Body-frame SE(2) helpers shared by simulation and controllers."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def body_error(
    x: float,
    y: float,
    yaw: float,
    target_x: float,
    target_y: float,
    target_yaw: float = 0.0,
) -> tuple[float, float, float]:
    """Return target error as forward, left, and yaw components."""

    dx, dy = target_x - x, target_y - y
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
        wrap_angle(target_yaw - yaw),
    )


def cumulative_matrix(horizon: int, dimension: int) -> np.ndarray:
    if horizon <= 0 or dimension <= 0:
        raise ValueError("horizon and dimension must be positive")
    return np.kron(np.tril(np.ones((horizon, horizon))), np.eye(dimension))


def difference_matrix(horizon: int, dimension: int) -> np.ndarray:
    if horizon <= 0 or dimension <= 0:
        raise ValueError("horizon and dimension must be positive")
    if horizon == 1:
        return np.zeros((0, dimension))
    result = np.zeros(((horizon - 1) * dimension, horizon * dimension))
    identity = np.eye(dimension)
    for index in range(horizon - 1):
        start = index * dimension
        result[start : start + dimension, start : start + dimension] = -identity
        result[start : start + dimension, start + dimension : start + 2 * dimension] = identity
    return result


def step_body_velocity(
    pose: np.ndarray | tuple[float, float, float],
    command: np.ndarray | tuple[float, float, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a body velocity and return (new_pose, measured_increment)."""

    state = np.asarray(pose, dtype=float)
    velocity = np.asarray(command, dtype=float)
    if (
        state.shape != (3,)
        or velocity.shape != (3,)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(velocity))
    ):
        raise ValueError("pose and command must be finite vectors of shape (3,)")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    x, y, yaw = state
    vx, vy, yaw_rate = velocity
    next_yaw = wrap_angle(yaw + yaw_rate * dt)
    next_pose = np.array(
        [
            x + (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt,
            y + (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt,
            next_yaw,
        ],
    )
    delta = next_pose[:2] - state[:2]
    increment = np.array(
        [
            math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
            -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1],
            wrap_angle(next_yaw - yaw),
        ],
    )
    return next_pose, increment
