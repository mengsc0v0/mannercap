"""Run the local, deterministic Gate 2/E0 synthesis preflight.

This entry point only reads a draft JSONL request manifest and local YAML
contracts.  It writes pending/unknown planning rows for G0--G3 and explicit
``api_not_configured`` skip rows for M.  It never calls a model, API,
controller, simulator, or final-audit artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis_v2.preflight import (  # noqa: E402
    PreflightError,
    run_preflight,
    write_preflight_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="draft request_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON preflight report output",
    )
    parser.add_argument(
        "--method-config-dir",
        type=Path,
        default=ROOT / "configs" / "generation_methods",
        help="directory containing the five method YAML files",
    )
    parser.add_argument(
        "--constraint-registry",
        type=Path,
        default=ROOT / "configs" / "constraint_registry.yaml",
        help="development constraint registry YAML",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_preflight(
            args.manifest,
            method_config_dir=args.method_config_dir,
            registry_path=args.constraint_registry,
        )
        write_preflight_report(args.output, report)
    except PreflightError as exc:
        _build_parser().error(str(exc))
        return 2  # pragma: no cover - argparse.error raises SystemExit

    summary = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "request_count": report["manifest"]["request_count"],
        "method_request_lanes": len(report["records"]),
        "preflight_methods": report["method_suite"]["preflight_methods"],
        "skipped": report["method_suite"]["skipped"],
        "charged_attempts": report["attempt_ledger"]["charged_attempts"],
        "output": str(args.output.expanduser().resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
