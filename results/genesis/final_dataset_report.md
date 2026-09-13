# Final expert imitation dataset report

This report covers collection, canonical processing, validation, provenance, coverage, and episode-disjoint splitting. BC/student training, student evaluation, PPO refinement, DeePC integration, and expert modification were not run.

## Final classification

`EXPERT_IMITATION_DATASET_PASS`

## Dataset size

- Attempted episodes: `120`
- Accepted episodes: `120`
- Rejected episodes: `0`
- Accepted samples: `240000`
- Accepted duration: `4800.000 s`
- Train: `96` episodes / `192000` samples
- Validation: `12` episodes / `24000` samples
- Test: `12` episodes / `24000` samples

## Frozen dimensions and ranges

- Student state: `[240000, 45]`
- Desired command: `[240000, 3]`
- Student input: `[240000, 48]`
- Expert action: `[240000, 12]`
- Action range: `[-2.11742425, 2.72153664]`
- Fraction outside `[-1,1]`: `22.5666%`
- q-target reconstruction max error: `0.000e+00`

## Safety and integrity

- Falls: `0`
- Base contacts: `0`
- Actual joint-limit violations: `0`
- Torque-limit violations: `0`
- NaN/Inf flags: `0`
- Temporal integrity: `PASS`
- Privilege leakage audit: `PASS`
- Reproducibility spot check: `PASS`

## Coverage

The command bank includes standing, forward, backward characterization, lateral, pure yaw, forward+yaw arcs, forward+lateral, full mixed commands, starts/stops, sign changes, and multi-segment transitions. See `reports/command_coverage.md`, `reports/split_coverage.md`, and `plots/`.

- Canonical dataset SHA256: `7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99`
- Dataset hash manifest SHA256: `a9659d01c75a4f5762128a8676a3ae9dff8bd6b4d1498a874a0757a4c5e86e60`

EXPERT_IMITATION_DATASET_PASS
