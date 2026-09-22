# Gate 2 独立物理校准候选包

报告 ID：`gate2.calibration_candidate_report.20260921`
候选包状态：`candidate_ready=false`
物理资格：`unknown`；阈值冻结：`false`；晋升：`false`

本报告只读本地报告、哈希清单、纯测量契约以及 `reports/gate2/calibration_raw_20260921` 下已取回的原始 payload；每个 payload 先按原 SHA-256 清单核验。 汇总数字仍不被当作原始传感器、时间轴或噪声；contact_pair 保持未校准。

## 输入证据

- manifest：`reports/gate2/calibration_input_manifest_20260921.json`，canonical SHA-256 `1faea0844a1292558da06a79864bd1421a8c511fea360a122391680853632842`
- candidate registry：`reports/gate2/constraint_registry_candidate_20260921.json`，registry hash `699219e0db6ee045bbce7ba14fefbcda9a5747f9bbcc9332e5b4e12aaa4b962e`
- 本地 raw measurement 条目：`3`
- raw contract audit：`unknown`，records=`480`，valid=`480`

## 结论

### `body_wrench_layout_and_units` — partially_recomputed

- 判定：`unknown`
- 结论：The contracts distinguish MuJoCo body wrench torque (N*m) and force (N), and reject mixed six-dimensional norms.
- 限制：The retrieved controlled-load rows verify vector shape, frame, sign transform, units, and explicit body/body-geom binding for 480 rows; object/contact attribution, resolution/noise, and physical qualification remain open.
- 证据：
  - `gate1_force_measurement_spec` — `reports/gate1/force_measurement_spec.md` — SHA-256 `caecc1979e4c64bbd9d8d12b7e88901b9f71b0a1e2ea283ff13d6302fc434862`
  - `force_calibration_contract` — `synthesis_v2/force_calibration.py` — SHA-256 `15da0e79d06be71aad4d7512f7ba1d895733abac6cda7dd71abed49bd97bed98`
  - `contact_force_contract` — `synthesis_v2/contact_force_contract.py` — SHA-256 `c5557efa5aa041b3f9eebb4c134364f2a9558c537b9a234c50632c40ee6cb2b2`
  - `raw:gate1_force_evidence_sha:controlled_wrench_calibration.json` — `reports/gate2/calibration_raw_20260921/controlled_wrench_calibration.json` — SHA-256 `71dbe4139c52d09b78da375806e4b07ebbfe3d897ba4ee7422a7859f44cb841f`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — SHA-256 `59058f26c9b54532e6798fffbdc92e8f1430deb56012cf89ba68d1cd5328c061`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — `reports/gate2/calibration_raw_20260921/gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`

### `known_load_response` — raw_rows_recomputed_partial

- 判定：`unknown`
- 结论：SHA-verified controlled-load rows were parsed locally and paired within each case; 240 baseline/loaded row pairs were recomputed. The body response remains a candidate input and does not promote any threshold.
- 限制：Remaining contract gaps are recorded from the parsed rows: independent_reference_not_declared, object_id_missing_on_some_body_rows
- 证据：
  - `gate1_wipe_controlled_load_results` — `reports/gate1/wipe_controlled_load_results.md` — SHA-256 `35595d22f8d8c35f387a5cf30d6c34dbf7b307ead00c9f78931324764e46b9fa`
  - `gate1_wipe_controlled_load_sha` — `reports/gate1/wipe_controlled_load_20260917.sha256` — SHA-256 `ff3067c2e18686ca6191b64285d9773c413a6c5529bb6a7107f686bde723a432`
  - `gate1_force_calibration_results` — `reports/gate1/force_calibration_results.md` — SHA-256 `366fdca43855598ed75b5bb4cea8cebfa010e4688107ee72fcb2a90bf8d54077`
  - `gate1_force_evidence_sha` — `reports/gate1/evidence_manifest_attempt03.sha256` — SHA-256 `d99e58e9786b6b6d6746617bb4fae9949da65e41df58ce63688cc49dada40493`
  - `raw:gate1_force_evidence_sha:controlled_wrench_calibration.json` — `reports/gate2/calibration_raw_20260921/controlled_wrench_calibration.json` — SHA-256 `71dbe4139c52d09b78da375806e4b07ebbfe3d897ba4ee7422a7859f44cb841f`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — SHA-256 `59058f26c9b54532e6798fffbdc92e8f1430deb56012cf89ba68d1cd5328c061`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — `reports/gate2/calibration_raw_20260921/gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`

### `contact_pair_and_object_attribution` — partially_observed

- 判定：`unknown`
- 结论：Hash-verified representative contact rows expose contact frame, geom/body IDs, vectors, and sample time, but contact force direction attribution and calibration remain unknown.
- 限制：92 of 92 representative contact rows passed structural checks; the payload does not expose full transition rows or an independent known-load mapping, and all contact calibration remains unverified.
- 证据：
  - `gate1_force_calibration_protocol` — `reports/gate1/force_calibration_protocol.md` — SHA-256 `5f303adbc0503872f4321d01357f68cd95007df2b0ce69c1796355e0414f1e85`
  - `gate1_force_calibration_results` — `reports/gate1/force_calibration_results.md` — SHA-256 `366fdca43855598ed75b5bb4cea8cebfa010e4688107ee72fcb2a90bf8d54077`
  - `gate1_wipe_contact_dense_results` — `reports/gate1/wipe_contact_dense_results.md` — SHA-256 `504fe11a2677138fabb448bc4cf3b347d6f3f5f823369a7bb11f4b6f2bf40af0`
  - `gate1_wipe_contact_dense_sha` — `reports/gate1/wipe_contact_dense_20260918.sha256` — SHA-256 `9a8e72c69346faeb0005763e37639f5199a27a6a9699048b09578702f670e07b`
  - `raw:gate1_force_evidence_sha:lift/transitions.jsonl` — `lift/transitions.jsonl` — SHA-256 `a2fdb94cacc72c2c08575e5557a88320be6eb8f237e4a22395e081eeabc26d2f`
  - `raw:gate1_force_evidence_sha:stack/transitions.jsonl` — `stack/transitions.jsonl` — SHA-256 `286052031c4c918d3190eedd731c4826b9099bd58c442b1ebe6a0663ec4932aa`
  - `raw:gate1_force_evidence_sha:force_diagnostic.json` — `reports/gate2/calibration_raw_20260921/force_diagnostic.json` — SHA-256 `05c1ec967e8e4e7ad8e99037a59cc6aa80179213df20281dc701314b4eb7274b`
  - `raw:gate1_force_evidence_sha:controlled_wrench_calibration.json` — `reports/gate2/calibration_raw_20260921/controlled_wrench_calibration.json` — SHA-256 `71dbe4139c52d09b78da375806e4b07ebbfe3d897ba4ee7422a7859f44cb841f`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_under/transitions.jsonl` — `gate1_wipe_calibration_under/transitions.jsonl` — SHA-256 `fd52ea560afa6dcbe03961e17700aa2f7927113c07ccc5f18f22babe28b0754d`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_nominal/transitions.jsonl` — `gate1_wipe_calibration_nominal/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_over/transitions.jsonl` — `gate1_wipe_calibration_over/transitions.jsonl` — SHA-256 `25e4936c68d6aca92fed1d414261f4e2d3da7cae43a31d10e9c27afeda85d2f8`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `9f4180905af6e42578967a9acc4b9af3a6c4303417b13a4f60282b87980db661`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-001-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-001-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-002-seed-42-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-002-seed-42-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `25e4936c68d6aca92fed1d414261f4e2d3da7cae43a31d10e9c27afeda85d2f8`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-003-seed-43-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-003-seed-43-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `33e32b78fe15d86de5243a541dcef9b3c4e8d7b57ed4b699dde704c238e19a3f`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-004-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-004-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `bb463088494e6fb1610173429b7be3d3246d37cd3ebf7ac2bd8a906c6be9906a`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-005-seed-43-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-005-seed-43-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `7810ffac01a2ac35f71ef75850bb76747e22f8f9d3a08d9efdff435049475fbe`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-006-seed-44-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-006-seed-44-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `31faa122c75ea19fc6fe2afe80b67904e96a8feb1d5f276fc7308de937b53487`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-007-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-007-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `426ad0879a43c34ee44291b3ce92a4df98e008a507d7cae2decf6be391bc75fb`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-008-seed-44-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-008-seed-44-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `aa90846c194aef33dcf4a7c8214193f725dd024d9fbb5aaf75770d37e248fcd6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `9f4180905af6e42578967a9acc4b9af3a6c4303417b13a4f60282b87980db661`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-001-seed-42-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-001-seed-42-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `228edcbedeb641b8d21c78a57292f8730edc985f91bcb9d3054a1f24103bbb7a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-002-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-002-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-003-seed-42-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-003-seed-42-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `6d7efe82217112cfe831e1dede110ad8987e7b431f7bf2981609d7016d9e90f7`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-004-seed-42-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-004-seed-42-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `01b9c73d6a0544d61afc5d49871e101b3b2ea1ca1e13d84059879a5708e8e6c2`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-005-seed-43-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-005-seed-43-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `33e32b78fe15d86de5243a541dcef9b3c4e8d7b57ed4b699dde704c238e19a3f`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-006-seed-43-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-006-seed-43-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `d672d856de37cfd2dbc4c4219205ebddee2efd463e475f7db9b43f6a92976778`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-007-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-007-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `bb463088494e6fb1610173429b7be3d3246d37cd3ebf7ac2bd8a906c6be9906a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-008-seed-43-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-008-seed-43-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `62ef940e6204cefa4cc7ec44e680e00d7195175e52077509b982d1ea7f330878`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-009-seed-43-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-009-seed-43-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `88b81b34175f6ff8ed01df08ae3516e792c0f9823dd710e4d8876d20db79e6e6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-010-seed-44-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-010-seed-44-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `31faa122c75ea19fc6fe2afe80b67904e96a8feb1d5f276fc7308de937b53487`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-011-seed-44-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-011-seed-44-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `6ac6151cd732421a3240e978dcff5cdca2ab4bad14f6db2da3cfdd93b8ed1922`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-012-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-012-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `426ad0879a43c34ee44291b3ce92a4df98e008a507d7cae2decf6be391bc75fb`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-013-seed-44-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-013-seed-44-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `665f7a523b4e8ae8f5a03f0e85483c20e21738e23b8c55a097c23b734c376145`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-014-seed-44-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-014-seed-44-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `6c1556173c6ffbadd6bae593395e15f6bcdb2b757f911d679bebd45abaf981c9`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-015-seed-45-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-015-seed-45-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `eb9e8c9a9040a1a7164d867ffcee03cb6bb293392b3463672bc4c531882d60cb`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-016-seed-45-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-016-seed-45-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `ea339df42350eb71936a8a5fa8e97496534c9a060040ac6c6d0c7905eeec0f3a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-017-seed-45-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-017-seed-45-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `24f010831c23dd625f8eebd983b1b58e39096a25a5b0a86d011c8e3d571d84c6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-018-seed-45-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-018-seed-45-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `212806b978b32ad3b190f2c68e06802066d3337ae3a9d51d3c7ad414e8c1e47b`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-019-seed-45-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-019-seed-45-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `35c12e4ca2095323dae2d40ccf6442bee329d3d707ebbd435473b15ab7e39740`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-020-seed-46-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-020-seed-46-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `cc68aea81ad87a0482c61b201c7f19143a514c0a5ca59e9783c0c9f36e084b2f`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-021-seed-46-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-021-seed-46-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `1cb2aca37d59052d566977452c292d2f9ff5140ed910c58384e20b2217c8dba4`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-022-seed-46-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-022-seed-46-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `b3396b3b8c77e4a312215178bfb52e3ede3f5815423d69da2d0ba94bdd38974e`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-023-seed-46-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-023-seed-46-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `ee19462e9dd0e90e6961f02b3168802ae32eda4d1d63b5a7eb38ab9ade89cc31`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-024-seed-46-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-024-seed-46-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `47b4a4175416d8269fc309353c8db541856cd1ffe88de7237848692295ef04f6`

### `wipe_geometry_vs_force` — geometry_only

- 判定：`unknown`
- 结论：The Wipe 0--4 mm grid can be referenced as a development geometry/task response, not as a known-force calibration or safety threshold.
- 限制：Surface penetration is a control/geometry input; it is not a known Newton load, and local raw transitions are absent.
- 证据：
  - `gate1_wipe_contact_dense_results` — `reports/gate1/wipe_contact_dense_results.md` — SHA-256 `504fe11a2677138fabb448bc4cf3b347d6f3f5f823369a7bb11f4b6f2bf40af0`
  - `gate1_wipe_contact_grid_results` — `reports/gate1/wipe_contact_grid_results.md` — SHA-256 `6ab29e6802d9174ed71fc318681cc739176a9272d34e77d3498dbda957478625`
  - `gate1_wipe_contact_dense_sha` — `reports/gate1/wipe_contact_dense_20260918.sha256` — SHA-256 `9a8e72c69346faeb0005763e37639f5199a27a6a9699048b09578702f670e07b`

### `time_sampling_and_resolution` — partially_recomputed

- 判定：`unknown`
- 结论：Raw sample timestamps were checked within case/reset/load streams; adjacent dt was derived only between explicit timestamps, with each stream's first dt left unknown.
- 限制：timestamp_dt=pass; streams=24; derived_dt_rows=456; gaps=0; resolution_noise=unknown.
- 证据：
  - `trajectory_contract` — `synthesis_v2/trajectory_contract.py` — SHA-256 `e2a30072df23ac00945967fe3c212bd8d465e0b2e35cf5588799b47fd9a14221`
  - `gate1_force_calibration_protocol` — `reports/gate1/force_calibration_protocol.md` — SHA-256 `5f303adbc0503872f4321d01357f68cd95007df2b0ce69c1796355e0414f1e85`
  - `gate1_force_evidence_sha` — `reports/gate1/evidence_manifest_attempt03.sha256` — SHA-256 `d99e58e9786b6b6d6746617bb4fae9949da65e41df58ce63688cc49dada40493`
  - `raw:gate1_force_evidence_sha:lift/transitions.jsonl` — `lift/transitions.jsonl` — SHA-256 `a2fdb94cacc72c2c08575e5557a88320be6eb8f237e4a22395e081eeabc26d2f`
  - `raw:gate1_force_evidence_sha:stack/transitions.jsonl` — `stack/transitions.jsonl` — SHA-256 `286052031c4c918d3190eedd731c4826b9099bd58c442b1ebe6a0663ec4932aa`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — SHA-256 `59058f26c9b54532e6798fffbdc92e8f1430deb56012cf89ba68d1cd5328c061`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — `reports/gate2/calibration_raw_20260921/gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_under/transitions.jsonl` — `gate1_wipe_calibration_under/transitions.jsonl` — SHA-256 `fd52ea560afa6dcbe03961e17700aa2f7927113c07ccc5f18f22babe28b0754d`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_nominal/transitions.jsonl` — `gate1_wipe_calibration_nominal/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_calibration_sha:gate1_wipe_calibration_over/transitions.jsonl` — `gate1_wipe_calibration_over/transitions.jsonl` — SHA-256 `25e4936c68d6aca92fed1d414261f4e2d3da7cae43a31d10e9c27afeda85d2f8`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `9f4180905af6e42578967a9acc4b9af3a6c4303417b13a4f60282b87980db661`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-001-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-001-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-002-seed-42-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-002-seed-42-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `25e4936c68d6aca92fed1d414261f4e2d3da7cae43a31d10e9c27afeda85d2f8`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-003-seed-43-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-003-seed-43-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `33e32b78fe15d86de5243a541dcef9b3c4e8d7b57ed4b699dde704c238e19a3f`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-004-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-004-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `bb463088494e6fb1610173429b7be3d3246d37cd3ebf7ac2bd8a906c6be9906a`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-005-seed-43-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-005-seed-43-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `7810ffac01a2ac35f71ef75850bb76747e22f8f9d3a08d9efdff435049475fbe`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-006-seed-44-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-006-seed-44-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `31faa122c75ea19fc6fe2afe80b67904e96a8feb1d5f276fc7308de937b53487`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-007-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-007-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `426ad0879a43c34ee44291b3ce92a4df98e008a507d7cae2decf6be391bc75fb`
  - `raw:gate1_wipe_contact_grid_sha:./cases/case-008-seed-44-penetration-0p01m/probe_output/transitions.jsonl` — `./cases/case-008-seed-44-penetration-0p01m/probe_output/transitions.jsonl` — SHA-256 `aa90846c194aef33dcf4a7c8214193f725dd024d9fbb5aaf75770d37e248fcd6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-000-seed-42-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `9f4180905af6e42578967a9acc4b9af3a6c4303417b13a4f60282b87980db661`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-001-seed-42-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-001-seed-42-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `228edcbedeb641b8d21c78a57292f8730edc985f91bcb9d3054a1f24103bbb7a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-002-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-002-seed-42-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `a54e501fc2daf8c894a66cdd6a733224c3aee9eaa8ed4e734d2e5cfa07a5dc73`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-003-seed-42-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-003-seed-42-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `6d7efe82217112cfe831e1dede110ad8987e7b431f7bf2981609d7016d9e90f7`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-004-seed-42-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-004-seed-42-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `01b9c73d6a0544d61afc5d49871e101b3b2ea1ca1e13d84059879a5708e8e6c2`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-005-seed-43-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-005-seed-43-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `33e32b78fe15d86de5243a541dcef9b3c4e8d7b57ed4b699dde704c238e19a3f`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-006-seed-43-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-006-seed-43-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `d672d856de37cfd2dbc4c4219205ebddee2efd463e475f7db9b43f6a92976778`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-007-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-007-seed-43-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `bb463088494e6fb1610173429b7be3d3246d37cd3ebf7ac2bd8a906c6be9906a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-008-seed-43-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-008-seed-43-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `62ef940e6204cefa4cc7ec44e680e00d7195175e52077509b982d1ea7f330878`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-009-seed-43-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-009-seed-43-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `88b81b34175f6ff8ed01df08ae3516e792c0f9823dd710e4d8876d20db79e6e6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-010-seed-44-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-010-seed-44-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `31faa122c75ea19fc6fe2afe80b67904e96a8feb1d5f276fc7308de937b53487`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-011-seed-44-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-011-seed-44-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `6ac6151cd732421a3240e978dcff5cdca2ab4bad14f6db2da3cfdd93b8ed1922`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-012-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-012-seed-44-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `426ad0879a43c34ee44291b3ce92a4df98e008a507d7cae2decf6be391bc75fb`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-013-seed-44-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-013-seed-44-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `665f7a523b4e8ae8f5a03f0e85483c20e21738e23b8c55a097c23b734c376145`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-014-seed-44-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-014-seed-44-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `6c1556173c6ffbadd6bae593395e15f6bcdb2b757f911d679bebd45abaf981c9`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-015-seed-45-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-015-seed-45-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `eb9e8c9a9040a1a7164d867ffcee03cb6bb293392b3463672bc4c531882d60cb`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-016-seed-45-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-016-seed-45-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `ea339df42350eb71936a8a5fa8e97496534c9a060040ac6c6d0c7905eeec0f3a`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-017-seed-45-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-017-seed-45-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `24f010831c23dd625f8eebd983b1b58e39096a25a5b0a86d011c8e3d571d84c6`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-018-seed-45-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-018-seed-45-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `212806b978b32ad3b190f2c68e06802066d3337ae3a9d51d3c7ad414e8c1e47b`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-019-seed-45-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-019-seed-45-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `35c12e4ca2095323dae2d40ccf6442bee329d3d707ebbd435473b15ab7e39740`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-020-seed-46-penetration-0m/probe_output/transitions.jsonl` — `./cases/case-020-seed-46-penetration-0m/probe_output/transitions.jsonl` — SHA-256 `cc68aea81ad87a0482c61b201c7f19143a514c0a5ca59e9783c0c9f36e084b2f`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-021-seed-46-penetration-0p001m/probe_output/transitions.jsonl` — `./cases/case-021-seed-46-penetration-0p001m/probe_output/transitions.jsonl` — SHA-256 `1cb2aca37d59052d566977452c292d2f9ff5140ed910c58384e20b2217c8dba4`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-022-seed-46-penetration-0p002m/probe_output/transitions.jsonl` — `./cases/case-022-seed-46-penetration-0p002m/probe_output/transitions.jsonl` — SHA-256 `b3396b3b8c77e4a312215178bfb52e3ede3f5815423d69da2d0ba94bdd38974e`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-023-seed-46-penetration-0p003m/probe_output/transitions.jsonl` — `./cases/case-023-seed-46-penetration-0p003m/probe_output/transitions.jsonl` — SHA-256 `ee19462e9dd0e90e6961f02b3168802ae32eda4d1d63b5a7eb38ab9ade89cc31`
  - `raw:gate1_wipe_contact_dense_sha:./cases/case-024-seed-46-penetration-0p004m/probe_output/transitions.jsonl` — `./cases/case-024-seed-46-penetration-0p004m/probe_output/transitions.jsonl` — SHA-256 `47b4a4175416d8269fc309353c8db541856cd1ffe88de7237848692295ef04f6`

### `independent_calibration_boundary` — not_ready

- 判定：`unknown`
- 结论：No independent known-load reference and no verified contact_pair calibration permit this candidate to be promoted to a physical qualification.
- 限制：The method output is explicitly excluded; independent_reference remains false and contact_pair calibration is intentionally unverified. All thresholds remain null/pending.
- 证据：
  - `gate1_constraint_registry_spec` — `reports/gate1/constraint_registry_spec.md` — SHA-256 `c1eeb59fefd16a71455686795cb60e0560e7ab3920b4eaa5fa22227a78ce25b9`
  - `checked_in_constraint_registry` — `configs/constraint_registry.yaml` — SHA-256 `b3a64bc221628d06232a5b844b95afd8fac7bb4182818ebb2ec6f41006829e6b`
  - `five_method_audit` — `reports/gate2/five_method_audit_20260921_v2.json` — SHA-256 `cca28cc7056f8e1f59a44b3242ccf04bb6c21c1300408d61f6c69f4c5421b5e9`
  - `raw:gate1_force_evidence_sha:force_diagnostic.json` — `reports/gate2/calibration_raw_20260921/force_diagnostic.json` — SHA-256 `05c1ec967e8e4e7ad8e99037a59cc6aa80179213df20281dc701314b4eb7274b`
  - `raw:gate1_force_evidence_sha:controlled_wrench_calibration.json` — `reports/gate2/calibration_raw_20260921/controlled_wrench_calibration.json` — SHA-256 `71dbe4139c52d09b78da375806e4b07ebbfe3d897ba4ee7422a7859f44cb841f`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917/controlled_load_calibration.json` — SHA-256 `59058f26c9b54532e6798fffbdc92e8f1430deb56012cf89ba68d1cd5328c061`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — `gate1_wipe_controlled_load_20260917_attempt02/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`
  - `raw:gate1_wipe_controlled_load_sha:gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — `reports/gate2/calibration_raw_20260921/gate1_wipe_controlled_load_20260917_attempt03/controlled_load_calibration.json` — SHA-256 `e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48`

## 下一实验必须补齐的输入

- `raw_controlled_load_payloads`：Retain the SHA-verified controlled-load JSON used here and retrieve any additional calibration cases or raw streams named by the manifests before parsing.
- `known_load_and_application_metadata`：For each sample record known force/torque in N/N*m, world direction, application point, site/body frame, sign convention, and baseline pairing.
- `raw_contact_rows`：For a future contact_pair audit, record ncon/contact geom IDs and mj_contactForce force/torque vectors per pair; keep cfrc_ext body aggregate separate. This package leaves contact_pair uncalibrated.
- `timing_and_validity`：Retain the within-stream timestamp audit and add control step, physics substep, reset/stage IDs, invalid spans, and explicit sampling resolution/noise; derived dt here never assumes a fixed step.
- `repeatability_and_noise`：Repeat no-contact and at least two known-load conditions from matched snapshots and record resolution/noise metadata; no formal threshold is set by this candidate package.
- `independent_calibration_split`：Freeze task/seed source initial-state hashes and lineage for a calibration split independent of method outputs; report object/config coverage without claiming object generalization.

## 边界

candidate registry 保持 `development_pending`，所有 lower/upper/tolerance 为 null；本包不设正式阈值、不把 geometry penetration 改名为力，且 contact_pair 不因 body-channel 或诊断数据而自动校准。
