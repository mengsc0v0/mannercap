# LA4VLA student development attempt03

Attempt03 completed the bounded action-head development check against the frozen 8-source seed42 train bundle and the existing 8-source seed43 development holdout. The runner passed the exact QC report SHA and plan SHA guards, resolved both dataset roots from their configs, and reported no source-binding mismatches.

The run used 100 optimizer steps from the original step-500 checkpoint, batch size 8, learning rate `1e-4`, seed `20260921`, one final checkpoint, and no model selection. It made no API calls, recaptured no data, and ran no simulator or closed-loop evaluation. The 6 GiB per-process cap was applied on shared low-utilization GPU3 after the registered read-only preflight. Together with the separately preserved attempt02 development run, this records 200 optimizer steps; only the two training wall times are known (`548.8265 + 227.1302 = 775.9567` seconds). The old startup failure duration is unknown, and capture/setup time is not included in that sum, so this is not a total end-to-end cost.

Fixed-noise flow loss decreased from `0.2558394913` to `0.1608582083` on train and from `0.2530226707` to `0.1609562170` on validation. The real `model.predict_action(actions_gt=None)->action_head.get_action` ODE used the same noise seed (`20261121`) for both passes. The reported 7D values are denormalized adapter control-input units after reusing the train-only normalization statistics. They are not errors expressed in meters or radians and must not be interpreted as physical pose or way-execution errors. Overall denormalized 7D metrics were:

| split | initial MAE | initial MSE | final MAE | final MSE |
|---|---:|---:|---:|---:|
| train | 0.126568 | 0.169106 | 0.067654 | 0.061321 |
| validation | 0.123125 | 0.163065 | 0.067148 | 0.060165 |

The run took 227.13 seconds and used 1,811,971,519 bytes of new tracked output. The final checkpoint SHA-256 is `61810299a72aa829f5542c7eb8b4d2d5587eeb274faae3925c429935fd88b9c6`.

This remains a single-seed, small development observation with `promotion_allowed=false`, `generalization_claim_allowed=false`, and `closed_loop_claim_allowed=false`; it does not establish way learning or physical/closed-loop performance. Attempt02 remains preserved and separately marked as a protocol deviation caused by its two-source train bundle; its metrics are not merged into this result.

Detailed JSON and hash inventory: `la4vla_student_dev_training_20260922_attempt03.json`, `la4vla_student_dev_attempt03_result_manifest_20260922.json`, `la4vla_student_dev_attempt03_plan_20260922.json`, `la4vla_student_dev_attempt03_resource_revision_20260922.json`, and `la4vla_student_dev_parquet_sha256_20260922.txt`.
