"""Classify cached M request/candidate differences without rerunning anything.

This tool reads the checked-in five-method semantic audit and produces a small
M-only report.  It never changes the original request, proposal, or audit
record, and it does not call a model, simulator, or remote service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthesis_v2.semantic_audit import classify_request_candidate_semantics


class SemanticAnalysisError(ValueError):
    """Raised when the source audit is incomplete or malformed."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAnalysisError(message)


def analyze(audit: Mapping[str, Any]) -> dict[str, Any]:
    _require(audit.get("schema") == "mannercap-e0-reconciliation-v1", "unexpected audit schema")
    records = audit.get("phase_semantic_audits")
    _require(isinstance(records, Mapping), "phase semantic audits are missing")
    m_records = {str(key): value for key, value in records.items() if "-M-" in str(key)}
    _require(len(m_records) == 8, f"expected 8 cached M records, got {len(m_records)}")

    rows: list[dict[str, Any]] = []
    difference_counts = {
        "representation_difference": 0,
        "substantive_conflict": 0,
        "unknown": 0,
    }
    overall_counts = {
        "match": 0,
        "representation_difference": 0,
        "substantive_conflict": 0,
        "unknown": 0,
    }
    for candidate_id, record in sorted(m_records.items()):
        _require(isinstance(record, Mapping), f"{candidate_id}: malformed audit record")
        source_diff = record.get("request_semantic_diff")
        _require(isinstance(source_diff, Mapping), f"{candidate_id}: missing request semantic diff")
        request_spec = source_diff.get("requested_spec")
        candidate_spec = source_diff.get("candidate_spec")
        _require(isinstance(request_spec, Mapping), f"{candidate_id}: missing requested_spec")
        _require(isinstance(candidate_spec, Mapping), f"{candidate_id}: missing candidate_spec")
        classified = classify_request_candidate_semantics(
            {"requested_manner": request_spec}, candidate_spec
        )
        overall = str(classified["overall"])
        overall_counts[overall] += 1
        for difference in classified["differences"]:
            difference_counts[difference["classification"]] += 1
        phase_scope_counts: dict[str, int] = {}
        for phase in record.get("phases", {}).values():
            scope = str(phase.get("scope"))
            phase_scope_counts[scope] = phase_scope_counts.get(scope, 0) + 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "request_spec": request_spec,
                "candidate_spec": candidate_spec,
                "classification": {
                    "overall": classified["overall"],
                    "differences": classified["differences"],
                    "contract": classified["contract"],
                },
                "requested_target_phases": source_diff.get("requested_target_phases", []),
                "candidate_target_phases": record.get("candidate_target_phases", []),
                "candidate_non_target_phases": record.get("candidate_non_target_phases", []),
                "execution_semantics_valid": record.get("valid") is True,
                "phase_scope_counts": phase_scope_counts,
                "manner_qualification": record.get("manner_qualification", "unknown"),
                "physical_interpretation": record.get(
                    "physical_interpretation", "descriptive_execution_semantics_only"
                ),
            }
        )

    return {
        "schema": "mannercap-m-request-candidate-semantic-analysis-v1",
        "source_audit_schema": audit["schema"],
        "source_plan_sha256": audit.get("plan_sha256"),
        "source_audit_valid": audit.get("valid") is True,
        "cached_m_records": len(rows),
        "overall_counts": overall_counts,
        "difference_counts": difference_counts,
        "rows": rows,
        "api_calls": 0,
        "simulator_calls": 0,
        "promotion_allowed": False,
        "manner_qualification": "unknown",
        "physical_interpretation": "request_candidate_semantics_only; no physical failure inferred",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    result = analyze(audit)
    payload = dict(result)
    payload["source_audit_sha256"] = hashlib.sha256(args.audit.read_bytes()).hexdigest()
    payload["report_sha256"] = _digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(result["rows"]), "report_sha256": payload["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
