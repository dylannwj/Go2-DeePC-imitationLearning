#!/usr/bin/env python3
"""Verify independent reload determinism for the validation-selected student."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learned_execution.linear_bc_common import DATASET_PATH, INPUT_DIM, REVISION_ROOT, load_dataset, make_model, sha256_file, write_json  # noqa: E402


def load_once(torch: object, checkpoint_path: Path, inputs: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(torch)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(inputs)).cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    selected = json.loads((revision_root / "manifests/selected_checkpoint.json").read_text(encoding="utf-8"))["selected"]
    checkpoint_path = Path(selected["candidate_path"])
    data = load_dataset(args.dataset.resolve())
    fixed_rows = np.linspace(0, len(data["student_input"]) - 1, 100, dtype=np.int64)
    inputs = np.asarray(data["student_input"][fixed_rows], dtype=np.float32)
    import torch

    torch.set_num_threads(1)
    first = load_once(torch, checkpoint_path, inputs)
    second = load_once(torch, checkpoint_path, inputs)
    max_abs_difference = float(np.max(np.abs(first - second)))
    result = {
        "schema": "linear-output-student-reload-determinism-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_count": int(len(inputs)),
        "input_dimension": INPUT_DIM,
        "max_absolute_difference": max_abs_difference,
        "exact_equal": bool(np.array_equal(first, second)),
        "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
        "classification": "STUDENT_RELOAD_DETERMINISM_PASS" if max_abs_difference == 0.0 and np.array_equal(first, second) else "STUDENT_RELOAD_DETERMINISM_BLOCKED",
    }
    write_json(revision_root / "manifests/reload_determinism.json", result)
    lines = [
        "# Student reload determinism",
        "",
        f"`{result['classification']}`",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint SHA256: `{result['checkpoint_sha256']}`",
        f"- fixed input count: `{result['input_count']}`",
        f"- input dimension: `{result['input_dimension']}`",
        f"- maximum absolute difference: `{result['max_absolute_difference']:.9g}`",
        f"- exact tensor equality: `{result['exact_equal']}`",
        "",
        "The checkpoint was loaded into two independently constructed model instances and evaluated on the same fixed 100 raw 48-D inputs.",
    ]
    (revision_root / "reports/student_reload_determinism.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] == "STUDENT_RELOAD_DETERMINISM_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
