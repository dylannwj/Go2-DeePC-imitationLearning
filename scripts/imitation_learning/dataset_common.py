#!/usr/bin/env python3
"""Shared deterministic storage and provenance helpers for the expert dataset.

This module intentionally depends only on the project environment's standard
library and NumPy.  In particular, it does not install or import a new data
format dependency.  The deterministic NPZ writer fixes ZIP timestamps so the
canonical dataset hash is stable across repeated writes of identical arrays.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/imitation"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash relative names, sizes, and file contents in a directory."""

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(child.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
        digest.update(b"\n")
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def runtime_payload() -> dict[str, Any]:
    try:
        import genesis as gs

        genesis_version = str(getattr(gs, "__version__", "unknown"))
        genesis_python = str(Path(gs.__file__).resolve())
    except Exception as exc:  # pragma: no cover - only missing runtime
        genesis_version = "unavailable"
        genesis_python = f"{type(exc).__name__}: {exc}"
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "genesis_version": genesis_version,
        "genesis_python": genesis_python,
        "project_revision": git_value("rev-parse", "HEAD"),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write an NPZ whose ZIP metadata is independent of wall-clock time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for name in arrays:
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ field name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    os.replace(temporary, path)


def metadata_array(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(canonical_json(dict(payload)))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def metadata_from_arrays(arrays: Mapping[str, np.ndarray], key: str = "metadata_json") -> dict[str, Any]:
    if key not in arrays:
        raise KeyError(f"{path_name(arrays)} has no {key}")
    value = arrays[key]
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar string array, got {value.shape}")
    return json.loads(str(value.item()))


def path_name(value: Any) -> str:
    return str(value) if isinstance(value, (str, Path)) else "NPZ archive"


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
