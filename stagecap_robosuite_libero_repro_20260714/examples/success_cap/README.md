# Successful CAP Code Examples

These files are raw successful `code.py` artifacts copied from recent Stage-CAP
runs. They are included as references for collaborators who want to understand
the style of generated CAP code, stage boundaries, `STAGE_STATUS` reporting,
and verification/repair logic.

Selection rules:

- Copied from runs whose path records `reward_1.000_taskcompleted_1`.
- Prefer recent `mimo-v2.5` Stage-CAP outputs.
- Exclude obvious old shortcut folders such as `fastpath`, `finalizer`, and
  `privfinal`.
- Audited by keyword scan for direct simulator mutation patterns such as
  `env.sim`, `sim.data`, `sim.model`, `qpos`, `qvel`, `set_state`, and
  `set_qpos`; no such direct state-edit pattern was found in the included
  files.

Important boundary:

These examples may use privileged readonly diagnostics and high-level robot
skills, because our target here is simulation training-data construction. They
should not be presented as clean policy-evaluation code.

## Included Examples

| File | Task | Main behavior shown |
| --- | --- | --- |
| `libero_goal_00_open_middle_drawer_success_cap.py` | Open the middle drawer of the cabinet | Drawer handle grounding, sleeve/hook attempt loop, open-fraction verification, structured repair feedback |
| `sfb1_02_cabinet_bowl_close_drawer_success_cap.py` | Pick a bowl from/open around a cabinet drawer and close the drawer | Drawer clearance guard, bowl pick/lift, placement verification, drawer close verification |
| `libero_object_00_alphabet_soup_basket_success_cap.py` | Pick alphabet soup and place it in the basket | Object/target grounding, pick/lift verification, basket placement |
| `sfb1_00_stove_moka_stage_success_cap.py` | Turn on stove and put moka pot on it | Button/stove grounding, stove action verification, object placement |
| `libero_goal_05_push_plate_front_stove_success_cap.py` | Push the plate to the front of the stove | Push-to-region stage and relation verification |
| `libero_goal_09_wine_bottle_rack_success_cap.py` | Put the wine bottle on the rack | Target grounding, grasp/transport/place verification |

For reproducible experiments, use the runner scripts in `scripts/` rather than
executing these files manually.
