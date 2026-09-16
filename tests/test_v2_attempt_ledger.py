"""Gate 1 tests for the bounded candidate-attempt ledger."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from synthesis_v2.attempt_ledger import (
    AttemptLedger,
    AttemptOutcome,
    LedgerError,
    RequestState,
    ResourceUsage,
    attempt_id_for,
    write_attempt_ledger,
)


REQUEST = {
    "request_id": "req_demo_01",
    "task": "lift",
    "split": "development",
    "initial_state": {"snapshot_ref": "state-01"},
    "raw_language": "lift the cube slowly",
    "requested_constraints": {"raw": {"speed": "as-requested"}},
}


def _resources(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "physics_substeps": 12,
        "api_calls": 1,
        "api_tokens": 25,
        "cpu_time_s": 0.2,
        "gpu_time_s": 0.0,
        "wall_time_s": 0.3,
        "storage_bytes": 1024,
    }
    result.update(overrides)
    return result


def _record_kwargs(outcome: str | AttemptOutcome, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "outcome": outcome,
        "proposal_source": "offline-test-proposer",
        "candidate_plan": {"stage_instance_id": "lift", "plan": "opaque"},
        "call_record": {"service": "none", "recorded": True},
        "resources": _resources(),
        "reason": f"explicit {str(outcome)} outcome",
    }
    result.update(overrides)
    return result


class AttemptLedgerContractTests(unittest.TestCase):
    def test_resource_vector_requires_all_nonnegative_dimensions(self):
        usage = ResourceUsage.from_mapping(_resources())
        self.assertEqual(usage.physics_substeps, 12)
        self.assertEqual(usage.api_calls, 1)
        self.assertEqual(usage.api_tokens, 25)
        self.assertEqual(usage.to_dict()["storage_bytes"], 1024)

        for field in (
            "physics_substeps",
            "api_calls",
            "api_tokens",
            "storage_bytes",
        ):
            values = _resources(**{field: -1})
            with self.assertRaisesRegex(LedgerError, "non-negative"):
                ResourceUsage.from_mapping(values)

        with self.assertRaisesRegex(LedgerError, "missing required"):
            ResourceUsage.from_mapping({})
        with self.assertRaises(LedgerError):
            ResourceUsage.from_mapping(_resources(cpu_time_s=float("nan")))

    def test_every_candidate_outcome_gets_a_unique_attempt_id_and_resources(self):
        ledger = AttemptLedger([REQUEST], max_attempts=6)
        request_id = REQUEST["request_id"]
        outcomes = (
            AttemptOutcome.PARSE_FAILURE,
            AttemptOutcome.INVALID,
            AttemptOutcome.EXECUTION_FAILURE,
            AttemptOutcome.VALIDATION_FAILURE,
            AttemptOutcome.DUPLICATE,
            AttemptOutcome.ACCEPTED,
        )
        records = []
        for outcome in outcomes:
            kwargs = _record_kwargs(outcome)
            if outcome == AttemptOutcome.PARSE_FAILURE:
                kwargs["candidate_plan"] = None
            if outcome == AttemptOutcome.DUPLICATE:
                kwargs["duplicate_of_attempt_id"] = "attempt-from-other-request"
            records.append(ledger.record_attempt(request_id, **kwargs))

        self.assertEqual([record.attempt_index for record in records], list(range(1, 7)))
        self.assertEqual(len({record.attempt_id for record in records}), 6)
        self.assertTrue(all(record.attempt_id.startswith("attempt_") for record in records))
        self.assertEqual(ledger.state(request_id), RequestState.ACCEPTED)
        self.assertEqual(len(ledger.records()), 6)
        for record in records:
            row = record.to_dict()
            self.assertEqual(row["attempt_id"], record.attempt_id)
            self.assertEqual(set(row["resources"]), {
                "physics_substeps",
                "api_calls",
                "api_tokens",
                "cpu_time_s",
                "gpu_time_s",
                "wall_time_s",
                "storage_bytes",
            })

    def test_max_attempts_is_shared_across_all_failure_kinds(self):
        ledger = AttemptLedger([REQUEST], max_attempts=3)
        request_id = REQUEST["request_id"]
        for outcome in (
            AttemptOutcome.PARSE_FAILURE,
            AttemptOutcome.INVALID,
            AttemptOutcome.EXECUTION_FAILURE,
        ):
            ledger.record_attempt(request_id, **_record_kwargs(outcome, candidate_plan=None))

        self.assertEqual(ledger.attempt_count(request_id), 3)
        self.assertEqual(ledger.remaining_attempts(request_id), 0)
        self.assertEqual(ledger.state(request_id), RequestState.BUDGET_EXHAUSTED)
        with self.assertRaisesRegex(LedgerError, "exhausted max_attempts"):
            ledger.record_attempt(
                request_id,
                **_record_kwargs(AttemptOutcome.VALIDATION_FAILURE, candidate_plan=None),
            )

    def test_acceptance_is_terminal_and_does_not_allow_hidden_followup_retry(self):
        ledger = AttemptLedger([REQUEST], max_attempts=3)
        request_id = REQUEST["request_id"]
        accepted = ledger.record_attempt(
            request_id, **_record_kwargs(AttemptOutcome.ACCEPTED)
        )
        self.assertEqual(accepted.request_state_after, RequestState.ACCEPTED.value)
        with self.assertRaisesRegex(LedgerError, "accepted"):
            ledger.record_attempt(
                request_id,
                **_record_kwargs(AttemptOutcome.EXECUTION_FAILURE, candidate_plan=None),
            )
        self.assertEqual(ledger.attempt_count(request_id), 1)

    def test_original_request_snapshot_is_copied_and_mutation_is_detected(self):
        source = copy.deepcopy(REQUEST)
        expected = copy.deepcopy(source)
        ledger = AttemptLedger([source], max_attempts=2)
        source["requested_constraints"]["raw"]["speed"] = "caller-mutated"
        source["initial_state"]["snapshot_ref"] = "caller-mutated"

        record = ledger.record_attempt(REQUEST["request_id"], **_record_kwargs("invalid"))
        self.assertEqual(record.request_snapshot, expected)
        self.assertEqual(record.to_dict()["request_snapshot"], expected)

        changed = copy.deepcopy(expected)
        changed["requested_constraints"]["raw"]["speed"] = "changed-after-register"
        with self.assertRaisesRegex(LedgerError, "changed after registration"):
            ledger.record_attempt(
                REQUEST["request_id"],
                **_record_kwargs("validation_failure", request_snapshot=changed),
            )

    def test_missing_required_attempt_fields_and_duplicate_reference_are_rejected(self):
        ledger = AttemptLedger([REQUEST], max_attempts=2)
        request_id = REQUEST["request_id"]
        with self.assertRaisesRegex(LedgerError, "outcome/status"):
            ledger.record_attempt(
                request_id,
                proposal_source="test",
                candidate_plan=None,
                call_record={},
                resources=_resources(),
                reason="missing outcome",
            )
        with self.assertRaisesRegex(LedgerError, "call_record"):
            ledger.record_attempt(
                request_id,
                **_record_kwargs("invalid", call_record=None),
            )
        with self.assertRaisesRegex(LedgerError, "duplicate outcome"):
            ledger.record_attempt(
                request_id,
                **_record_kwargs(AttemptOutcome.DUPLICATE),
            )
        with self.assertRaisesRegex(LedgerError, "accepted outcome"):
            ledger.record_attempt(
                request_id,
                **_record_kwargs(AttemptOutcome.ACCEPTED, candidate_plan=None),
            )

    def test_attempt_and_jsonl_ids_are_deterministic(self):
        self.assertEqual(attempt_id_for("req-x", 1), attempt_id_for("req-x", 1))
        self.assertNotEqual(attempt_id_for("req-x", 1), attempt_id_for("req-x", 2))
        ledger = AttemptLedger([REQUEST], max_attempts=2)
        ledger.record_attempt(REQUEST["request_id"], **_record_kwargs("parse_failure", candidate_plan=None))
        ledger.record_attempt(REQUEST["request_id"], **_record_kwargs("invalid", candidate_plan=None))

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger" / "attempt_ledger.jsonl"
            write_attempt_ledger(output, ledger)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["attempt_id"] for row in rows))
        self.assertEqual([row["attempt_index"] for row in rows], [1, 2])

    def test_status_alias_is_canonicalized_and_conflicts_are_rejected(self):
        ledger = AttemptLedger([REQUEST], max_attempts=2)
        request_id = REQUEST["request_id"]
        record = ledger.record_attempt(
            request_id,
            status="parse_error",
            proposal_source="test",
            candidate_plan=None,
            call_record={},
            resources=_resources(),
            reason="malformed response",
        )
        self.assertEqual(record.outcome, AttemptOutcome.PARSE_FAILURE.value)

        with self.assertRaisesRegex(LedgerError, "disagree"):
            conflict = _record_kwargs("invalid")
            conflict["status"] = "accepted"
            ledger.record_attempt(
                request_id,
                **conflict,
            )


if __name__ == "__main__":  # pragma: no cover - convenient local invocation
    unittest.main()
