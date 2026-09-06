# Package Note

Start with `README_ROBOSUITE_STAGECAP_ZH.md`.

This reproduction package contains the StageCaP implementation for all seven
CapX RoboSuite task families and the LIBERO 40-task StageCaP pipeline. It
includes the exact tracked source trees for RoboSuite 1.5.1, LIBERO-PRO,
LIBERO's RoboSuite 1.4.0 fork, and optional SAM3 support. The only local
third-party modification is recorded under `third_party_patches/`.

The package intentionally excludes virtual environments, API keys, model
weights, Git history, caches, paper PDFs, rollout videos, and large experiment
logs. Compact JSON result reports are under `repro_results/robosuite/`.

Use the runtime-specific environment scripts before each experiment:

```bash
source scripts/env_robosuite_stagecap.sh  # RoboSuite 1.5.1
source scripts/env_libero_stagecap.sh     # LIBERO + RoboSuite 1.4.0
```

See `VALIDATION_REPORT_20260714.md` for the tests and simulator smoke run made
from the staged archive tree. `PACKAGE_MANIFEST.sha256` is the authoritative
per-file integrity manifest.

See `ROBOSUITE_FAIR_COMPARISON_20260714.md` for the privileged Agent0 protocol,
the corrected labels for the historical API ablation, and the seven-task smoke
comparison.
