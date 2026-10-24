"""Versioned, budgeted, logged JSON calls to the language model (design v3 §2.8 / §8.1).

Everything the model is asked is addressed by ``(model, prompt_version, system, user)``; that tuple is
the cache key and the ledger key, so a run can be replayed without new calls and the cost of a run can
be recomputed afterwards.  The client never interprets content: parsing and validation belong to the
callers (``language_compiler``), so a malformed answer is a logged failure, not a silent default.

Credentials come from the environment only (``~/.config/mannercap/llm.env``); nothing here writes a key
to disk, to the ledger or to the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

LLM_INTERFACE_VERSION = "llm_interface.v1"


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would exceed the configured budget."""


@dataclass(frozen=True)
class LLMConfig:
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_calls: int = 32
    timeout_s: float = 120.0
    cache_dir: Optional[str] = None
    ledger_path: Optional[str] = None

    @staticmethod
    def from_env(**over: Any) -> "LLMConfig":
        cfg = LLMConfig(model=os.environ.get("MANNERCAP_LLM_MODEL", ""),
                        base_url=os.environ.get("AIHUBMIX_BASE_URL", ""))
        return replace(cfg, **over) if over else cfg

    def identity(self) -> dict[str, Any]:
        """Everything that can change an answer -- recorded with every result."""
        return {"interface": LLM_INTERFACE_VERSION, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_output_tokens": self.max_output_tokens}


@dataclass
class CallRecord:
    prompt_version: str
    request_hash: str
    status: str                      # ok | cached | parse_error | api_error | budget
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_version": self.prompt_version, "request_hash": self.request_hash,
                "status": self.status, "latency_s": round(self.latency_s, 3), "usage": self.usage,
                "error": self.error, "identity": self.identity}


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode()); h.update(b"\x00")
    return h.hexdigest()


def extract_json(text: str) -> Any:
    """Parse the first JSON object in ``text``; raises ``ValueError`` with the raw text on failure."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        s = s.split("\n", 1)[1] if s.lstrip().lower().startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"no JSON object in response: {text[:400]!r}")
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}): {s[i:j + 1][:400]!r}") from exc


class LLMClient:
    """Calls the model for one JSON answer at a time, with cache, budget and ledger."""

    def __init__(self, config: Optional[LLMConfig] = None, *, offline: bool = False) -> None:
        self.config = config or LLMConfig.from_env()
        self.offline = bool(offline)
        self.calls = 0
        self.records: list[CallRecord] = []
        self._client = None

    # ---- accounting -------------------------------------------------------
    def spend(self) -> dict[str, Any]:
        used = [r for r in self.records if r.status in ("ok", "api_error", "parse_error")]
        tok = lambda k: sum(int(r.usage.get(k, 0) or 0) for r in used)
        return {"calls": len(used), "cached": sum(r.status == "cached" for r in self.records),
                "failed": sum(r.status in ("api_error", "parse_error") for r in self.records),
                "latency_s": round(sum(r.latency_s for r in self.records), 3),
                "prompt_tokens": tok("prompt_tokens"), "completion_tokens": tok("completion_tokens"),
                "total_tokens": tok("total_tokens"), "budget": self.config.max_calls}

    def _log(self, rec: CallRecord) -> CallRecord:
        rec.identity = self.config.identity()
        self.records.append(rec)
        if self.config.ledger_path:
            p = Path(self.config.ledger_path); p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
        return rec

    # ---- cache ------------------------------------------------------------
    def _cache_file(self, key: str) -> Optional[Path]:
        return Path(self.config.cache_dir) / f"{key}.json" if self.config.cache_dir else None

    # ---- call -------------------------------------------------------------
    def complete_json(self, system: str, user: str, *, prompt_version: str) -> tuple[Any, CallRecord]:
        """Return ``(parsed_json, record)``; raises on budget, API or parse failure (always logged)."""
        key = _hash(self.config.model, prompt_version, system, user)
        cf = self._cache_file(key)
        if cf is not None and cf.exists():
            return json.loads(cf.read_text())["response"], self._log(
                CallRecord(prompt_version, key, "cached"))
        if self.offline:
            raise BudgetExceeded(f"offline client has no cached answer for {key[:12]}")
        if self.calls >= self.config.max_calls:
            self._log(CallRecord(prompt_version, key, "budget", error="max_calls reached"))
            raise BudgetExceeded(f"max_calls={self.config.max_calls} reached")

        t0 = time.time()
        self.calls += 1
        try:
            text, usage = self._raw(system, user)
        except Exception as exc:
            self._log(CallRecord(prompt_version, key, "api_error", time.time() - t0,
                                 error=f"{type(exc).__name__}: {exc}"))
            raise
        latency = time.time() - t0
        try:
            obj = extract_json(text)
        except ValueError as exc:
            self._log(CallRecord(prompt_version, key, "parse_error", latency, usage, str(exc)))
            raise
        if cf is not None:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({"prompt_version": prompt_version, "response": obj,
                                      "identity": self.config.identity()}, indent=2, sort_keys=True))
        return obj, self._log(CallRecord(prompt_version, key, "ok", latency, usage))

    def _raw(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("AIHUBMIX_API_KEY", "")
            if not key:
                raise RuntimeError("AIHUBMIX_API_KEY not set (source ~/.config/mannercap/llm.env)")
            self._client = OpenAI(api_key=key, base_url=self.config.base_url or None,
                                  timeout=self.config.timeout_s)
        r = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.config.temperature, max_completion_tokens=self.config.max_output_tokens,
            response_format={"type": "json_object"})
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        return (r.choices[0].message.content or ""), usage
