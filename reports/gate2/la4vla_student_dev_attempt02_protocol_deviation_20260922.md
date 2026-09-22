# LA4VLA student development attempt02: protocol deviation

The original attempt02 execution report is retained unchanged. It completed 100 action-head steps and two ODE sampling passes, but its train config resolved to the early two-source bundle `gate2_la4vla_microset_packed_20260918_attempt02`, while the frozen QC contract required `gate2_d0_full_la4vla_microset_packed_20260918` with eight seed42 source episodes.

The train-side evidence is therefore a bounded development observation only. It is not a pass of the frozen plan and must not be used as a promotion, generalization, or closed-loop result. No additional training was started from this deviation.

The source-lineage audit found:

- Actual train manifest SHA-256: `7c25f92647194c265097961d419f20d66d427758abace8756c8c8f11530965e2`.
- Frozen QC train manifest SHA-256: `4482a57125bf834e5b37d56d64658d5a1af08fa5edb00643920ca8369ae714ab`.
- Actual train source count: 2 (`lift`, `stack`), with 32 windows per task and no manner, initial-state, or source-result hashes.
- Frozen train source count: 8, covering `lift` and `stack` across `fast`, `fast_approach_slow_contact`, `neutral`, and `slow`, with eight windows per source.
- Validation binding is exact: the packed validation manifest and QC capture manifest both have SHA-256 `1215a0944c3bfbf11739dfeb9175345b15eb4cf3ac840abfb78ac5310f1596a0`, with eight seed43 source episodes.

Every packed bundle was read with the server venv and pyarrow: 64 parquet files, 50 rows per file, 3,200 rows, finite 7D actions and 8D states. The action/state payload values of all 3,200 rows match their corresponding manifest rows at 1e-6 tolerance; task and episode indices also match. The existing loader smoke checked 128 inline image hashes per actual train/validation split with no mismatches. The complete parquet SHA-256 list is recorded in `la4vla_student_dev_parquet_sha256_20260922.txt`, with the row-value result in `la4vla_student_dev_parquet_value_audit_20260922.json`.

The runner now resolves the dataset root from each config and fails closed unless the manifest hash, source IDs, source count, task×manner pairs, source initial-state hashes, source-result hash count, window counts, metadata source count, exact QC report SHA-256, and QC plan SHA-256 match the frozen record. The focused regression suite passes 9 tests.

The original report remains `la4vla_student_dev_training_20260921_attempt02.json` (SHA-256 `e6c3da461de0a8de720730bbac1073a0c10c84eab4ca9b6325d8dbc00981135e`), and the old startup failure remains preserved (SHA-256 `5cf7029fe892e7da3e82d54e400f1b26050dc9f5067b958b29e3a20fd51de448`). Detailed binding evidence is in `la4vla_student_dev_attempt02_dataset_binding_audit_20260922.json`.
