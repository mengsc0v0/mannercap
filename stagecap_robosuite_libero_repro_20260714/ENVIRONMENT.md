# Stage-CAP Environment Setup

This package is meant to be reproducible for collaborators, while keeping the
handoff tarball lightweight. The code, prompt templates, KITE keyframe
extractor, scene-edit scripts, and data conversion utilities are included.
Large simulator assets, model weights, and API keys are not bundled.

## Tested Local Environment

The original machine uses:

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python
```

Observed working versions:

```text
Python 3.12
numpy 1.26.4
opencv-python / cv2 4.12.0
torch 2.8.0+cu128
tyro 1.0.6
open3d 0.19.0
viser 1.0.21
```

If you are on the original machine, use this environment directly:

```bash
cd /path/to/stage_cap_data_generation_handoff_release_20260628
cp .env.example .env
# Fill MIMO_API_KEY in .env.
source scripts/env_stagecap.sh
"${PYTHON_BIN}" scripts/check_environment.py
```

## Recreate A Collaborator Environment

Use Python `>=3.10,<3.13`. Python 3.12 is the tested version.

```bash
cd /path/to/stage_cap_data_generation_handoff_release_20260628
python3.12 -m venv .venv-libero
source .venv-libero/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements-stagecap-runtime.txt
python -m pip install -r requirements-kite-vendored.txt
```

Equivalent helper:

```bash
bash scripts/bootstrap_stagecap_env.sh
```

`requirements-kite-vendored.txt` intentionally contains only the minimal KITE
dependencies used by our failure-frame extractor. The full upstream KITE
requirements file is preserved at `third_party/kite/requirements.txt`, but it
is not needed for this Stage-CAP handoff workflow.

LIBERO and robosuite may require the same third-party sources used by CAP-X.
If they are not already installed in your environment, install or mount the
CAP-X third-party dependencies before running LIBERO tasks:

```text
capx/third_party/LIBERO-PRO
capx/third_party/robosuite
capx/third_party/sam3          # only needed for SAM3 perception mode
pyroki                         # needed for the PyRoKi motion service
```

The release package does not include these large third-party folders.

## Required Runtime Variables

Copy `.env.example` to `.env` and fill in the real values:

```bash
cp .env.example .env
```

Important fields:

```text
PYTHON_BIN       Python executable for the prepared LIBERO/CAP-X environment.
PYTHONPATH       Should include the release root.
MUJOCO_GL        Use `egl` for headless GPU rendering.
MIMO_BASE_URL    OpenAI-compatible endpoint for MIMO.
MIMO_API_KEY     API key for MIMO.
MIMO_MODEL       Recommended: mimo-v2.5.
```

For the point-prompt VLM backend, our current setup can reuse MIMO:

```text
MOLMO_BASE_URL=${MIMO_BASE_URL}
MOLMO_API_KEY=${MIMO_API_KEY}
MOLMO_MODEL=mimo-v2.5
```

## Local Services

The privileged data-generation path can run without SAM3 if
`--use-sam3-perception` is not enabled. The common service ports are:

```text
PyRoKi motion planning:  http://127.0.0.1:8116
GraspNet grasp service:  http://127.0.0.1:8115
SAM3 segmentation:       http://127.0.0.1:8114
```

Start a minimal motion service:

```bash
"${PYTHON_BIN}" capx/serving/launch_servers.py --profile minimal
```

Start the default perception and motion services:

```bash
"${PYTHON_BIN}" capx/serving/launch_servers.py --profile default
```

Equivalent helper:

```bash
bash scripts/launch_stagecap_services.sh minimal
bash scripts/launch_stagecap_services.sh default
```

If SAM3 is already deployed elsewhere, set:

```bash
export SAM3_SERVICE_URL=http://host:port
```

## Environment Check

Basic check:

```bash
source scripts/env_stagecap.sh
"${PYTHON_BIN}" scripts/check_environment.py
```

Strict check for real experiments:

```bash
"${PYTHON_BIN}" scripts/check_environment.py --require-mimo-key --strict-services
```

The checker verifies:

- Python imports for Stage-CAP, CAP-X, KITE, OpenCV, Torch, Open3D, etc.
- Optional LIBERO/robosuite imports.
- MIMO and perception environment variables.
- Local service port reachability.

## Smoke Test

After the check passes, run a small LIBERO smoke test:

```bash
TASK_IDS="0" \
SUITES="libero_object" \
ROUND_TIMEOUT_SEC=420 \
TOTAL_TRIALS=1 \
bash scripts/run_libero40_codex_boundaries_mimo_kite.sh
```

For training-data construction, use the privileged Stage-CAP mode described in
`README.md`. Keep generated privileged datasets separate from clean policy
evaluation results.
