"""Prepare public M prompts offline; never execute a model or simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gate2_e0_observe import plan
from synthesis_v2.llm_proposal_contract import build_prompt


def prepare(seed: int = 50) -> dict:
    observation = plan(seed, 3)
    prompts = [{"request_id": request["request_id"],
                "prompt": build_prompt(request)} for request in observation["requests"]]
    payload = {
        "schema": "mannercap-llm-boundary-preparation-v1",
        "status": "offline_prepared_execution_blocked",
        "observation_plan_sha256": observation["plan_sha256"],
        "prompts": prompts,
        "api_calls": 0,
        "provider": None,
        "verified_model_id": None,
        "endpoint": None,
        "temperature": None,
        "token_cap": None,
        "cost_cap": None,
        "cache_policy": "persist_request_response_and_usage_before_retry",
        "required_before_call": [
            "verify_completed_G0_G3_server_observation_and_accounting",
            "configure_provider_model_and_credentials_outside_git",
            "freeze_sampling_token_and_cost_caps",
            "connect_response_parser_and_per_attempt_call_ledger",
        ],
        "scope": "explicit_instruction_velocity_planning_only",
        "formal_thresholds_frozen": False,
        "promotion_allowed": False,
    }
    payload["content_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_count": len(result["prompts"]), "api_calls": 0,
                      "content_sha256": result["content_sha256"]}))
