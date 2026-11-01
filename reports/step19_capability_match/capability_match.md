| 任务 | 需要 | 现有共享能力覆盖 | 缺口 | 预计新增 | 本地示范 |
|---|---|---|---|---|---|
| T8 | pick_place x2; scene-region target; two same-type objects | pick_place; @site region target (T5); per-object targets (T4) | moka-pot grasp pose / handle geometry unknown; same-type objects: binding of 'moka_pot_1' vs '_2' in language; no local demo file | config entry + (optional) demo-derived grasp_reference; no new primitive | 无 |
| T6 | pick_place x2; object target (On plate); scene-region target | pick_place; @site region target (T5); mug-on-plate config path (T4) | mug in-hand drift unresolved in T4 (contact policy 0/8); pudding grasp pose; no local demo file | config entry; mug stage inherits T4's open problem | 无 |
| T3 | pick_place; push-close drawer primitive | pick_place (adapt) | drawer-closing primitive missing | new execution primitive | 无 |
| T9 | pick_place; door-close primitive | pick_place (adapt) | door-closing primitive missing | new execution primitive | 无 |
| T2 | knob/switch primitive; pick_place | pick_place (adapt) | knob primitive missing | new execution primitive | 无 |