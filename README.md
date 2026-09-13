# DeePC-Go2-Thesis

Clean thesis release for the DeePC Go2 experiments. The repository keeps the
selected physical evidence, Genesis benchmark evidence, final student
checkpoint, DeePC implementation, and reproduction scripts.

## Layout

- `src/deepc/` — retained DeePC geometry, Hankel, QP, and Genesis controller.
- `src/learned_execution/` — frozen student/runtime and champion adapter.
- `scripts/` — physical, imitation-learning, and Genesis entry points.
- `data/physical/` and `results/physical/` — selected physical evidence.
- `data/imitation/` and `configs/imitation/` — split metadata and BC contracts.
- `results/genesis/` — canonical 43-case Genesis evidence and audits.

## Quick checks

Run from the repository root with the release environment installed:

```bash
PYTHONPATH=src:. python3 -c "import deepc, learned_execution.contracts"
sha256sum models/go2_observable_student_bc_linear_v1.pt
```

The Genesis runtime requires Genesis 1.3.1 and its bundled Go2/plane assets;
the physical branch additionally requires ROS 2, Gurobi, and a mocap stream.
See [`docs/environment.md`](docs/environment.md).

## Deferred release assets

The 246,636,478-byte processed imitation dataset is intentionally not stored
in normal Git history. Download `expert_imitation_dataset_v1.npz` from the
GitHub Release asset into `data/imitation/` and verify it using the SHA-256
instructions in [`data/imitation/README.md`](data/imitation/README.md).
The expert checkpoint and external Genesis/MJLab assets remain separately
licensed/downloaded dependencies.
