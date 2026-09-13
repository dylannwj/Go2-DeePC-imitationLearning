#!/usr/bin/env python3
"""Train the frozen-dataset 48->256 ELU->256 ELU->12 linear student."""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learned_execution.linear_bc_common import (  # noqa: E402
    ACTION_DIM,
    DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    INPUT_DIM,
    INPUT_FEATURE_NAMES,
    Q_DEFAULT_CANONICAL,
    REVISION_ROOT,
    behavior_metrics,
    load_dataset,
    load_split_indices,
    make_model,
    raw_student_targets,
    sha256_file,
    transition_masks,
    write_csv,
    write_json,
)


SEEDS = (0, 1, 2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_norm(parameters: Any) -> float:
    import torch

    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().to(dtype=torch.float64).pow(2).sum()
    return float(torch.sqrt(squared).item())


def output_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "finite": bool(np.isfinite(array).all()),
    }


def predict(model: Any, inputs: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    import torch

    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.asarray(inputs[start : start + batch_size], dtype=np.float32))
            rows.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, ACTION_DIM), dtype=np.float32)


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    l2 = np.linalg.norm(error, axis=1)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "mean_action_l2": float(np.mean(l2)),
        "median_action_l2": float(np.median(l2)),
        "p95_action_l2": float(np.percentile(l2, 95.0)),
        "p99_action_l2": float(np.percentile(l2, 99.0)),
        "max_action_l2": float(np.max(l2)),
    }


def identity_normalization() -> dict[str, Any]:
    return {
        "kind": "identity",
        "input": {"mean": [0.0] * INPUT_DIM, "std": [1.0] * INPUT_DIM},
        "note": "raw 48-D student input; no train-set standardization",
    }


def train_seed(
    *,
    seed: int,
    inputs: np.ndarray,
    targets: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    val_transition: np.ndarray,
    dataset_sha: str,
    revision_root: Path,
    batch_size: int,
    max_epochs: int,
    patience: int,
    threads: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    torch.set_num_threads(max(1, threads))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cpu")
    model = make_model(torch).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = torch.nn.MSELoss()

    train_inputs = torch.from_numpy(inputs[train_rows].astype(np.float32, copy=False))
    train_targets = torch.from_numpy(targets[train_rows].astype(np.float32, copy=False))
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        drop_last=False,
    )

    initial_sample = inputs[train_rows[: min(4096, len(train_rows))]]
    initial_output = predict(model, initial_sample)
    initial_sanity = {
        "sample_count": int(len(initial_sample)),
        "all_output": output_stats(initial_output),
        "per_joint_mean": np.mean(initial_output, axis=0).tolist(),
        "per_joint_std": np.std(initial_output, axis=0).tolist(),
        "per_joint_min": np.min(initial_output, axis=0).tolist(),
        "per_joint_max": np.max(initial_output, axis=0).tolist(),
    }

    epoch_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, Any] | None = None
    training_finite = True
    max_pred_seen = -float("inf")
    min_pred_seen = float("inf")
    max_grad_seen = 0.0
    nan_inf_batches = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_sum = 0
        max_grad_epoch = 0.0
        min_pred_epoch = float("inf")
        max_pred_epoch = -float("inf")
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs.to(device))
            prediction_np = prediction.detach().cpu().numpy()
            min_pred_epoch = min(min_pred_epoch, float(np.min(prediction_np)))
            max_pred_epoch = max(max_pred_epoch, float(np.max(prediction_np)))
            max_pred_seen = max(max_pred_seen, max_pred_epoch)
            min_pred_seen = min(min_pred_seen, min_pred_epoch)
            if not np.isfinite(prediction_np).all():
                training_finite = False
                nan_inf_batches += 1
                raise FloatingPointError(f"seed {seed} epoch {epoch}: non-finite prediction")
            loss = criterion(prediction, batch_targets.to(device))
            if not bool(torch.isfinite(loss)):
                training_finite = False
                nan_inf_batches += 1
                raise FloatingPointError(f"seed {seed} epoch {epoch}: non-finite loss")
            loss.backward()
            current_grad = grad_norm(model.parameters())
            if not np.isfinite(current_grad):
                training_finite = False
                nan_inf_batches += 1
                raise FloatingPointError(f"seed {seed} epoch {epoch}: non-finite gradient norm")
            max_grad_epoch = max(max_grad_epoch, current_grad)
            max_grad_seen = max(max_grad_seen, current_grad)
            optimizer.step()
            batch_count = len(batch_inputs)
            loss_sum += float(loss.item()) * batch_count
            sample_sum += batch_count

        model.eval()
        val_prediction = predict(model, inputs[val_rows])
        val_scalar = scalar_metrics(val_prediction, targets[val_rows])
        val_behavior = behavior_metrics(
            val_prediction,
            targets[val_rows],
            data_commands_for_rows,
            val_transition,
        )
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_mse": loss_sum / max(1, sample_sum),
            "val_mse": val_scalar["mse"],
            "val_mae": val_scalar["mae"],
            "val_mean_action_l2": val_scalar["mean_action_l2"],
            "val_median_action_l2": val_scalar["median_action_l2"],
            "val_p95_action_l2": val_scalar["p95_action_l2"],
            "gradient_norm_max": max_grad_epoch,
            "predicted_action_min": min_pred_epoch,
            "predicted_action_max": max_pred_epoch,
            "finite": True,
        }
        epoch_rows.append(row)
        if val_scalar["mse"] < best_val:
            best_val = val_scalar["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_prediction = val_prediction.copy()
        else:
            epochs_without_improvement += 1
        print(
            f"seed={seed} epoch={epoch:03d} train_mse={row['train_mse']:.8f} "
            f"val_mse={row['val_mse']:.8f} val_mae={row['val_mae']:.8f} "
            f"grad_max={row['gradient_norm_max']:.5g} "
            f"pred=[{row['predicted_action_min']:.5g},{row['predicted_action_max']:.5g}]",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"seed {seed} did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    val_prediction = predict(model, inputs[val_rows])
    val_scalar = scalar_metrics(val_prediction, targets[val_rows])
    val_behavior = behavior_metrics(val_prediction, targets[val_rows], data_commands_for_rows, val_transition)

    checkpoint = {
        "schema": "observable_student_bc_linear_v1",
        "architecture": [INPUT_DIM, 256, 256, ACTION_DIM],
        "hidden_activation": "ELU",
        "output_activation": "linear",
        "output_semantics": "raw_action_canonical_genesis_order",
        "input_dimension": INPUT_DIM,
        "state_dimension": 45,
        "command_dimension": 3,
        "action_dimension": ACTION_DIM,
        "input_features": list(INPUT_FEATURE_NAMES_FOR_CHECKPOINT),
        "target_source": "expert_action_student_order",
        "normalization": identity_normalization(),
        "action_scale": [0.5] * ACTION_DIM,
        "q_default_canonical": Q_DEFAULT_CANONICAL.tolist(),
        "pd_kp": [20.0, 20.0, 40.0] * 4,
        "pd_kd": [1.0, 1.0, 2.0] * 4,
        "dataset_sha256": dataset_sha,
        "seed": seed,
        "data_loader_seed": seed,
        "device": str(device),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "training_script_sha256": sha256_file(Path(__file__)),
        "optimizer": "Adam",
        "learning_rate": 1.0e-3,
        "batch_size": batch_size,
        "maximum_epochs": max_epochs,
        "early_stopping_patience": patience,
        "selected_epoch": best_epoch,
        "model_state_dict": best_state,
        "validation_metrics": val_scalar,
        "initial_output_sanity": initial_sanity,
    }
    candidate_path = revision_root / "checkpoints/candidates" / f"seed_{seed}_best.pt"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, candidate_path)
    candidate_hash = sha256_file(candidate_path)

    write_csv(revision_root / "logs" / f"seed_{seed}_training.csv", epoch_rows)
    write_json(
        revision_root / "metrics" / f"seed_{seed}_validation.json",
        {
            "seed": seed,
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_hash,
            "selected_epoch": best_epoch,
            "epochs_completed": len(epoch_rows),
            "validation": val_scalar,
            "validation_by_behavior": val_behavior,
            "initial_output_sanity": initial_sanity,
            "training_safety": {
                "finite": training_finite,
                "nan_inf_batches": nan_inf_batches,
                "max_gradient_norm": max_grad_seen,
                "minimum_predicted_action_seen": min_pred_seen,
                "maximum_predicted_action_seen": max_pred_seen,
            },
        },
    )
    return {
        "seed": seed,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "selected_epoch": best_epoch,
        "epochs_completed": len(epoch_rows),
        "validation": val_scalar,
        "validation_by_behavior": val_behavior,
        "initial_output_sanity": initial_sanity,
        "training_safety": {
            "finite": training_finite,
            "nan_inf_batches": nan_inf_batches,
            "max_gradient_norm": max_grad_seen,
            "minimum_predicted_action_seen": min_pred_seen,
            "maximum_predicted_action_seen": max_pred_seen,
        },
    }


# These globals are set once by main and make the per-epoch call signature
# compact without hiding any data transformation or normalization.
data_commands_for_rows: np.ndarray
INPUT_FEATURE_NAMES_FOR_CHECKPOINT: tuple[str, ...]


def write_initial_sanity_report(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Linear student output initialization sanity",
        "",
        "The final layer is linear and uses standard framework initialization. No hand-tuned final bias was applied.",
        "",
        "| Seed | Samples | Mean | Std | Min | Max | Finite |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        stats = result["initial_output_sanity"]["all_output"]
        lines.append(
            f"| {result['seed']} | {result['initial_output_sanity']['sample_count']} | "
            f"{stats['mean']:.9g} | {stats['std']:.9g} | {stats['min']:.9g} | {stats['max']:.9g} | "
            f"{stats['finite']} |"
        )
    lines += [
        "",
        "The reported values are the untrained raw network outputs over a fixed sample of train observations. They are logged before any optimizer step.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global data_commands_for_rows, INPUT_FEATURE_NAMES_FOR_CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_epochs <= 0 or args.patience <= 0:
        raise SystemExit("batch size, max epochs, and patience must be positive")

    revision_root = args.revision_root.resolve()
    dataset_path = args.dataset.resolve()
    data = load_dataset(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit(f"frozen dataset hash mismatch: {dataset_sha}")
    preflight_path = revision_root / "manifests/revised_preflight.json"
    if not preflight_path.is_file():
        raise SystemExit("run preflight_linear.py before training")
    import json

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("classification") != "LINEAR_BC_PREFLIGHT_PASS":
        raise SystemExit("linear BC preflight did not pass")

    inputs = np.asarray(data["student_input"], dtype=np.float32)
    targets = raw_student_targets(data)
    train_rows = load_split_indices("train")
    val_rows = load_split_indices("val")
    transition_all, _ = transition_masks(data["episode_id"], data["step_index"], data["desired_command"])
    data_commands_for_rows = np.asarray(data["desired_command"][val_rows], dtype=np.float32)
    val_transition = transition_all[val_rows]
    INPUT_FEATURE_NAMES_FOR_CHECKPOINT = tuple(INPUT_FEATURE_NAMES)
    if len(INPUT_FEATURE_NAMES_FOR_CHECKPOINT) != INPUT_DIM:
        raise RuntimeError("input feature contract has the wrong dimension")

    results: list[dict[str, Any]] = []
    started = time.time()
    for seed in args.seeds:
        results.append(
            train_seed(
                seed=int(seed),
                inputs=inputs,
                targets=targets,
                train_rows=train_rows,
                val_rows=val_rows,
                val_transition=val_transition,
                dataset_sha=dataset_sha,
                revision_root=revision_root,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                threads=args.threads,
            )
        )
    summary = {
        "schema": "linear-output-bc-training-summary-v1",
        "generated_utc": utc_now(),
        "dataset_sha256": dataset_sha,
        "training_script_sha256": sha256_file(Path(__file__)),
        "seeds": [int(seed) for seed in args.seeds],
        "configuration": {
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "batch_size": args.batch_size,
            "maximum_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "input_preprocessing": "identity/raw",
            "loss": "MSE(student_action, expert_action_student_order)",
            "device": "cpu",
            "torch_threads": args.threads,
        },
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    write_json(revision_root / "manifests/training_summary.json", summary)
    write_initial_sanity_report(results, revision_root / "reports/initial_output_sanity.md")
    print(f"completed {len(results)} seeds in {summary['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
