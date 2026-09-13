# Final linear-output student BC report

STUDENT_BC_LINEAR_CLOSED_LOOP_PASS

ARCHITECTURE REVISION VALIDATED:
48 → 256 ELU → 256 ELU → 12 linear

The frozen dataset, teacher, action scale, PD gains, and observable student input contract were unchanged. The only hypothesis change was removal of the final tanh restriction.

## Frozen artifacts

- final checkpoint: `/home/dylan/Desktop/Go2Project/student_bc/linear_output_revision/checkpoints/final/go2_observable_student_bc_linear_v1.pt`
- final checkpoint SHA256: `6169c02e924dfb09078d9cba63e4071abd11a477ab2b104e92506df7ce0ed7cc`
- training script SHA256: `7aba76d1c96904ef2155025ed79e4743c80f7840d0e63bd716a46e22ef87f2c6`
- dataset SHA256: `7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99`
- seed / selected epoch: `2 / 98`
- Genesis: `1.3.1`

## Gate results

- `LINEAR_BC_PREFLIGHT_PASS`
- `LINEAR_BC_OFFLINE_PASS`
- `STUDENT_RELOAD_DETERMINISM_PASS`
- `STUDENT_BC_LINEAR_CLOSED_LOOP_PASS`

Closed-loop safety counts were zero for falls, base contacts, NaN/Inf, actual joint-limit violations, and torque-limit exceedances. Standing, forward, yaw, arc, lateral, transition, full-duration, and video gates passed across 33 student episodes. Raw neural actions were not clamped to `[-1,1]`; 60 raw q-target joint-limit clip events were logged separately by the physical safety layer.

Held-out test evidence is in `reports/test_results.md` and `metrics/test_metrics.csv`; reload evidence is in `reports/student_reload_determinism.md`; runtime state distribution evidence is in `reports/closed_loop_distribution_shift.md`.

## Required boundary

No PPO, DAgger, teacher fallback, action blending, policy switching, gait switching, or privileged state was added.

NEXT AUTHORIZED PHASE:
FREEZE FINAL STUDENT LOCOMOTION ENVELOPE
AND PREPARE DeePC INTEGRATION

DO NOT START AUTOMATICALLY.

STUDENT_BC_LINEAR_CLOSED_LOOP_PASS
