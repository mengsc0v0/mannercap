"""Bounded M proposal smoke with persistent call reservations and no retries."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gate2_e0_observe import plan
from synthesis_v2.llm_proposal_contract import build_prompt, parse_response


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def credentials(path):
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    keys = [x for x in lines if x.startswith("sk-")]
    urls = [x for x in lines if x.startswith("https://")]
    if len(keys) != 1 or len(urls) != 1:
        raise ValueError("Expected exactly one credential and one HTTPS endpoint")
    parts = urlsplit(urls[0])
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Endpoint must not contain userinfo, query, or fragment")
    return keys[0], urls[0].rstrip("/")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    key, endpoint = credentials(args.credentials)
    requests = plan(50, 1)["requests"][:args.limit]
    payloads = [{"model": args.model, "temperature": 0, "max_tokens": 512,
                 "stream": False, "enable_thinking": False,
                 "messages": [{"role": "user", "content": json.dumps(build_prompt(r))}]}
                for r in requests]
    manifest = {"schema": "mannercap-m-api-smoke-v1", "endpoint": endpoint,
                "model": args.model, "max_generation_calls": args.limit,
                "max_output_tokens_per_call": 512, "automatic_retries": 0,
                "requests": requests, "payloads": payloads,
                "scope": "development_explicit_velocity_planning", "promotion_allowed": False}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "plan.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Existing plan mismatch")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Output must be empty")
        write(target, manifest)
    if not args.run:
        print(json.dumps({"planned_calls": args.limit, "generation_calls": 0}))
        return
    opener = urllib.request.build_opener(NoRedirect)
    for index, (request, payload) in enumerate(zip(requests, payloads)):
        reservation = args.output / f"call_{index:02d}.json"
        if reservation.exists():
            raise RuntimeError("Call already reserved; inspect cached evidence before retry")
        record = {"request_id": request["request_id"], "model": args.model,
                  "status": "reserved_outcome_unknown", "attempt_index": 1,
                  "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                  "generation_calls_reserved": 1, "usage": None}
        # Exclusive creation occurs before network dispatch; crashes cannot cause hidden retries.
        with reservation.open("x", encoding="utf-8") as f:
            json.dump(record, f)
        started = time.monotonic()
        try:
            req = urllib.request.Request(endpoint + "/chat/completions",
                data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
                "Content-Type": "application/json"}, method="POST")
            with opener.open(req, timeout=45) as response:
                raw = response.read(1048576).decode()
            # Never persist a response that echoes the secret.
            if key in raw:
                raise ValueError("Credential echo detected")
            decoded = json.loads(raw)
            write(args.output / f"response_{index:02d}.json", decoded)
            record["usage"] = decoded.get("usage")
            record["returned_model"] = decoded.get("model")
            text = decoded["choices"][0]["message"]["content"]
            record["candidate_preset"] = parse_response(text, request["task"])
            record["status"] = "parsed"
        except urllib.error.HTTPError as exc:
            record.update(status="http_error", http_status=exc.code)
        except Exception as exc:
            # Exception messages and HTTP bodies can contain sensitive values.
            record.update(status="call_or_parse_error", error_type=type(exc).__name__)
        record["wall_s"] = time.monotonic() - started
        write(reservation, record)
        print(json.dumps({"index": index, "status": record["status"], "usage": record["usage"]}), flush=True)
        if record["status"] != "parsed":
            break


if __name__ == "__main__":
    main()
