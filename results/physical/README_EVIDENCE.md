# MIQP 3-input DeePC final-yaw evidence

Status: working satisfactorily.

Evidence target:
- 3-input MIQP DeePC using vx, vy, yaw_rate.
- UI supports right-click target B, mouse aim final heading arrow, left-click lock final yaw.
- Controller reaches A→B and corrects final yaw heading.
- Stale MIQP solve protection added so old solves cannot send commands after stop.
- Practical final yaw tolerance used.

Files preserved:
- deepc_ui_node_FINAL_YAW_UI.py
- miqp_online_receding_deepc_controller_FINAL_3INPUT_YAW.py
- deadzone_miqp_deepc_preview_solver_3INPUT_DOG2.py
- Hankel files and center correction report
- latest solver CSV/JSON
- ROS topic snapshots

Notes:
- Yaw-on-spot data exists in the Hankel in both directions.
- Full Hankel with too many columns caused slow/bad MIQP timeout behavior.
- Reduced/balanced column bank helped yaw-on-spot solve quickly.
