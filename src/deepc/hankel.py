"""Hankel construction and dataset boundary utilities.

The repository stores no research datasets. These functions operate on a
user-supplied sequence or CSV file and write an explicitly requested output.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _finite_matrix(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty 2-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def block_hankel(
    samples: np.ndarray | Sequence[Sequence[float]], depth: int, *, stride: int = 1
) -> np.ndarray:
    """Return a time-major block Hankel matrix.

    A window [x[k], ..., x[k + depth - 1]] is flattened in time-major order,
    matching the controller convention used by Up/Uf/Yp/Yf.
    """

    data = _finite_matrix(samples, "samples")
    if depth <= 0 or stride <= 0:
        raise ValueError("depth and stride must be positive")
    starts = range(0, len(data) - depth + 1, stride)
    columns = [data[start : start + depth].reshape(-1) for start in starts]
    if not columns:
        raise ValueError(f"not enough samples ({len(data)}) for depth {depth}")
    return np.stack(columns, axis=1)


def count_cross_reset_hankel_windows(
    trajectory_ids: Sequence[object] | np.ndarray, depth: int, *, stride: int = 1
) -> int:
    """Count depth windows that would cross an independent-trajectory boundary."""

    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1:
        raise ValueError(f"trajectory_ids must be one-dimensional, got {ids.shape}")
    if depth <= 0 or stride <= 0:
        raise ValueError("depth and stride must be positive")
    return sum(
        not np.all(ids[start : start + depth] == ids[start])
        for start in range(0, len(ids) - depth + 1, stride)
    )


def mosaic_block_hankel(
    samples: np.ndarray | Sequence[Sequence[float]],
    trajectory_ids: Sequence[object] | np.ndarray,
    depth: int,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Build a block-Hankel mosaic without crossing reset-separated trajectories.

    ``samples`` is a concatenation of independently reset trajectories and
    ``trajectory_ids`` identifies the owning trajectory of each row.  Mixed-ID
    windows are discarded before columns are assembled.  The provenance map
    records both the accepted window locations and the number of rejected
    boundary-crossing windows so callers can audit the construction.
    """

    data = _finite_matrix(samples, "samples")
    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1 or len(ids) != len(data):
        raise ValueError(
            "trajectory_ids must be one-dimensional and aligned with samples "
            f"(got {ids.shape} for {len(data)} samples)"
        )
    if depth <= 0 or stride <= 0:
        raise ValueError("depth and stride must be positive")
    if len(data) < depth:
        raise ValueError(f"not enough samples ({len(data)}) for depth {depth}")

    columns: list[np.ndarray] = []
    trajectory_ids_out: list[object] = []
    local_starts: list[int] = []
    global_starts: list[int] = []
    global_ends: list[int] = []
    run_start_by_index = np.zeros(len(ids), dtype=np.int64)
    current_run_start = 0
    for index, _trajectory_id in enumerate(ids):
        if index == 0 or not np.array_equal(ids[index], ids[index - 1]):
            current_run_start = index
        run_start_by_index[index] = current_run_start

    skipped = 0
    for start in range(0, len(data) - depth + 1, stride):
        window_ids = ids[start : start + depth]
        if not np.all(window_ids == window_ids[0]):
            skipped += 1
            continue
        key = window_ids[0].item() if isinstance(window_ids[0], np.generic) else window_ids[0]
        columns.append(data[start : start + depth].reshape(-1))
        trajectory_ids_out.append(key)
        local_starts.append(start - int(run_start_by_index[start]))
        global_starts.append(start)
        global_ends.append(start + depth)
    if not columns:
        raise ValueError(f"no trajectory contains enough samples for depth {depth}")
    provenance: dict[str, np.ndarray | int] = {
        "trajectory_id": np.asarray(trajectory_ids_out),
        "start_index": np.asarray(local_starts, dtype=np.int64),
        "global_start_index": np.asarray(global_starts, dtype=np.int64),
        "global_end_index_exclusive": np.asarray(global_ends, dtype=np.int64),
        "cross_reset_hankel_windows": 0,
        "rejected_cross_reset_windows": skipped,
    }
    return np.stack(columns, axis=1), provenance


def multi_trajectory_block_hankel(
    samples: np.ndarray | Sequence[Sequence[float]],
    trajectory_ids: Sequence[object] | np.ndarray,
    depth: int,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Compatibility name for :func:`mosaic_block_hankel`."""

    return mosaic_block_hankel(samples, trajectory_ids, depth, stride=stride)


def _as_label_array(labels: Sequence[str] | None, dimension: int, name: str) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"{name}_{index}" for index in range(dimension))
    result = tuple(str(label) for label in labels)
    if len(result) != dimension or len(set(result)) != dimension:
        raise ValueError(f"{name}_labels must contain {dimension} unique labels")
    return result


@dataclass(frozen=True)
class HankelData:
    """Validated DeePC past/future Hankel partition."""

    Up: np.ndarray
    Uf: np.ndarray
    Yp: np.ndarray
    Yf: np.ndarray
    T_ini: int
    N: int
    u_labels: tuple[str, ...] = ()
    y_labels: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        arrays = {
            "Up": np.asarray(self.Up, dtype=float),
            "Uf": np.asarray(self.Uf, dtype=float),
            "Yp": np.asarray(self.Yp, dtype=float),
            "Yf": np.asarray(self.Yf, dtype=float),
        }
        if self.T_ini <= 0 or self.N <= 0:
            raise ValueError("T_ini and N must be positive")
        for name, array in arrays.items():
            if array.ndim != 2 or array.shape[1] == 0:
                raise ValueError(f"{name} must be a nonempty 2-D array")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
            array = np.array(array, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (
            self.Up.shape[1] != self.Uf.shape[1]
            or self.Up.shape[1] != self.Yp.shape[1]
            or self.Up.shape[1] != self.Yf.shape[1]
        ):
            raise ValueError("all Hankel blocks must have the same column count")
        if self.Up.shape[0] % self.T_ini or self.Yp.shape[0] % self.T_ini:
            raise ValueError("past block row counts must be divisible by T_ini")
        if self.Uf.shape[0] % self.N or self.Yf.shape[0] % self.N:
            raise ValueError("future block row counts must be divisible by N")
        if self.u_dim != self.Up.shape[0] // self.T_ini or self.u_dim != self.Uf.shape[0] // self.N:
            raise ValueError("input dimensions are inconsistent")
        if self.y_dim != self.Yp.shape[0] // self.T_ini or self.y_dim != self.Yf.shape[0] // self.N:
            raise ValueError("output dimensions are inconsistent")
        object.__setattr__(
            self, "u_labels", _as_label_array(self.u_labels or None, self.u_dim, "u")
        )
        object.__setattr__(
            self, "y_labels", _as_label_array(self.y_labels or None, self.y_dim, "y")
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def columns(self) -> int:
        return self.Up.shape[1]

    @property
    def u_dim(self) -> int:
        return self.Up.shape[0] // self.T_ini

    @property
    def y_dim(self) -> int:
        return self.Yp.shape[0] // self.T_ini

    def truncate_horizon(self, horizon: int) -> HankelData:
        """Return the same data bank with a shorter future horizon."""

        if horizon <= 0 or horizon > self.N:
            raise ValueError(f"horizon must be in [1, {self.N}]")
        return HankelData(
            self.Up,
            self.Uf[: horizon * self.u_dim],
            self.Yp,
            self.Yf[: horizon * self.y_dim],
            self.T_ini,
            horizon,
            self.u_labels,
            self.y_labels,
            self.metadata,
        )


def build_hankel(
    input_sequences: Iterable[np.ndarray | Sequence[Sequence[float]]],
    output_sequences: Iterable[np.ndarray | Sequence[Sequence[float]]],
    *,
    T_ini: int,
    N: int,
    stride: int = 1,
    u_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> HankelData:
    """Build a Hankel bank from one or more independent episodes."""

    if T_ini <= 0 or N <= 0:
        raise ValueError("T_ini and N must be positive")
    input_list = list(input_sequences)
    output_list = list(output_sequences)
    if len(input_list) != len(output_list) or not input_list:
        raise ValueError("input_sequences and output_sequences must be nonempty and aligned")
    length = T_ini + N
    up_blocks, uf_blocks, yp_blocks, yf_blocks = [], [], [], []
    u_dim = y_dim = None
    for index, (inputs, outputs) in enumerate(zip(input_list, output_list, strict=True)):
        u = _finite_matrix(inputs, f"input sequence {index}")
        y = _finite_matrix(outputs, f"output sequence {index}")
        if len(u) != len(y):
            raise ValueError(f"sequence {index} input/output lengths differ")
        if u_dim is None:
            u_dim, y_dim = u.shape[1], y.shape[1]
        if (u.shape[1], y.shape[1]) != (u_dim, y_dim):
            raise ValueError("all sequences must have consistent dimensions")
        hu = block_hankel(u, length, stride=stride)
        hy = block_hankel(y, length, stride=stride)
        up_blocks.append(hu[: T_ini * u_dim])
        uf_blocks.append(hu[T_ini * u_dim :])
        yp_blocks.append(hy[: T_ini * y_dim])
        yf_blocks.append(hy[T_ini * y_dim :])
    return HankelData(
        np.concatenate(up_blocks, axis=1),
        np.concatenate(uf_blocks, axis=1),
        np.concatenate(yp_blocks, axis=1),
        np.concatenate(yf_blocks, axis=1),
        T_ini,
        N,
        _as_label_array(u_labels, int(u_dim), "u"),
        _as_label_array(y_labels, int(y_dim), "y"),
        metadata,
    )


def save_hankel(path: str | Path, hankel: HankelData) -> Path:
    """Save a compact, metadata-bearing Hankel package."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        Up=hankel.Up,
        Uf=hankel.Uf,
        Yp=hankel.Yp,
        Yf=hankel.Yf,
        T_ini=np.int64(hankel.T_ini),
        N=np.int64(hankel.N),
        u_labels=np.asarray(hankel.u_labels),
        y_labels=np.asarray(hankel.y_labels),
        metadata_json=json.dumps(hankel.metadata, sort_keys=True),
    )
    return destination


def _scalar_int(archive: Mapping[str, np.ndarray], name: str) -> int:
    value = np.asarray(archive[name]).reshape(-1)
    if len(value) != 1:
        raise ValueError(f"{name} must contain one scalar")
    return int(value[0])


def load_hankel(path: str | Path) -> HankelData:
    """Load and validate a Hankel package without allowing object pickles."""

    with np.load(path, allow_pickle=False) as archive:
        required = {"Up", "Uf", "Yp", "Yf", "T_ini", "N"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Hankel package is missing {missing}")
        metadata = {}
        if "metadata_json" in archive.files:
            raw = np.asarray(archive["metadata_json"]).reshape(-1)
            if len(raw) != 1:
                raise ValueError("metadata_json must contain one string")
            metadata = json.loads(str(raw[0]))
        u_labels = (
            tuple(str(value) for value in archive["u_labels"])
            if "u_labels" in archive.files
            else ()
        )
        y_labels = (
            tuple(str(value) for value in archive["y_labels"])
            if "y_labels" in archive.files
            else ()
        )
        return HankelData(
            archive["Up"],
            archive["Uf"],
            archive["Yp"],
            archive["Yf"],
            _scalar_int(archive, "T_ini"),
            _scalar_int(archive, "N"),
            u_labels,
            y_labels,
            metadata,
        )


def body_frame_increments(
    pose_xyyaw: np.ndarray | Sequence[Sequence[float]],
    *,
    yaw_offset_rad: float = 0.0,
) -> np.ndarray:
    """Convert world-frame pose samples to local [forward, left, yaw] increments."""

    pose = _finite_matrix(pose_xyyaw, "pose_xyyaw")
    if pose.shape[1] != 3 or len(pose) < 2:
        raise ValueError("pose_xyyaw must have shape (T, 3) with T >= 2")
    yaw = np.unwrap(pose[:, 2]) + float(yaw_offset_rad)
    delta = pose[1:, :2] - pose[:-1, :2]
    cosine, sine = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    forward = cosine * delta[:, 0] + sine * delta[:, 1]
    left = -sine * delta[:, 0] + cosine * delta[:, 1]
    yaw_delta = np.arctan2(
        np.sin(pose[1:, 2] - pose[:-1, 2]),
        np.cos(pose[1:, 2] - pose[:-1, 2]),
    )
    return np.column_stack((forward, left, yaw_delta))


def csv_to_sequences(
    path: str | Path,
    *,
    input_columns: Sequence[str],
    pose_columns: Sequence[str] = ("mocap_x_m", "mocap_y_m", "mocap_yaw"),
    yaw_offset_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one project CSV log and produce aligned DeePC input/output rows."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = tuple(pose_columns) + tuple(input_columns)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    pose = np.asarray([[float(row[column]) for column in pose_columns] for row in rows])
    commands = np.asarray([[float(row[column]) for column in input_columns] for row in rows])
    if len(commands) < 2:
        raise ValueError("CSV must contain at least two rows")
    return commands[:-1], body_frame_increments(pose, yaw_offset_rad=yaw_offset_rad)
