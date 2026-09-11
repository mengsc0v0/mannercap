"""Manner Parser — 用 LLM 把自然语言描述解析为 MannerSpec 物理约束参数。

Architecture:
    自然语言描述
        ↓  Gemini 3.7 Flash (structured JSON)
    MannerSpec(max_joint_vel, kp_max, grip_fraction_max, max_jerk, ...)
        ↓  物理层执行
    goto_pose_with_profile / set_impedance_stiffness / close_gripper_partial
        ↓  MannerValidator
    SATISFIED / NOT_SATISFIED + repair_hint

Usage::

    from capx.llm.manner_parser import LLMMannerParser

    parser = LLMMannerParser()           # reads AIHUBMIX_API_KEY from env
    spec = parser.parse("像捧着装满水的杯子那样把方块放到盒子上")
    print(spec)   # MannerSpec(max_joint_vel=0.04, kp_max=420, ...)

Env vars
--------
AIHUBMIX_API_KEY  :  API key for aihubmix.com  (required)
MANNER_PARSER_MODEL: override model  (default: gemini-3.7-flash)
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import asdict
from typing import Any

# ── lazy import: openai is optional dep ──────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from capx.envs.manner_validator import MannerSpec, PhaseVelocity, MANNER_PRESETS, compose_manner_spec

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_MODEL   = "gemini-3.7-flash"
_DEFAULT_BASEURL = "https://aihubmix.com/v1"
_FALLBACK_BASEURL = "https://api.inferera.com/v1"

# Physical parameter ranges (for prompt context)
_PARAM_GUIDE = """
Physical parameter reference (Franka Panda @ 20 Hz control):
  max_joint_vel     (rad/s)  : 0.02=极缓  0.06=慢  0.15=正常  0.30=快  0.50=极快
  min_joint_vel     (rad/s)  : None 表示无下限约束
  kp_max            (stiffness): 300=极柔  500=轻柔  900=正常  1200=偏硬  1500=最硬
  kp_min            (stiffness): None 表示无下限约束
  grip_fraction_max (0~1)    : 0.10=微触  0.25=轻握  0.50=半握  0.80=紧握  1.0=全力
  grip_fraction_min (0~1)    : None 表示无下限约束
  max_jerk          (rad/s³) : 0.5=极平滑  2.0=平滑  10=正常  None=不限

Rules:
  - "慢/轻/柔/缓/小心/稳" → max_joint_vel↓, kp_max↓, grip_fraction_max↓, max_jerk↓
  - "快/迅速/急/猛" → min_joint_vel↑, kp_min↑, grip_fraction_min↑
  - "用力/紧/牢/死死" → kp_min↑, grip_fraction_min↑
  - "平滑/丝滑/流畅" → max_jerk↓
  - 不提及某维度时，对应字段设为 null（不约束）

Phase-aware manner (optional, only when description clearly implies DIFFERENT
speed at DIFFERENT stages of the trajectory, e.g. "先快后慢"/"快速接近，轻轻
放下"/"先...再..."):
  - Output an additional top-level "phases" object (omit entirely, or leave as
    null, if the description describes ONE constant manner for the whole
    trajectory — do NOT invent phase variation that isn't implied).
  - Allowed phase names (use only the ones implied by the description; a task
    may only have some of these depending on what it does):
      approach   (机械臂靠近物体阶段)
      descend    (下降对准/贴合物体阶段)
      lift       (提起物体阶段)
      transport  (搬运/横移阶段)
      final      (最终放下/精细动作阶段)
      retreat    (完成后撤离阶段)
  - Each phase entry: {"max_joint_vel": <float|null>, "min_joint_vel": <float|null>}
    using the same rad/s scale as the flat max_joint_vel/min_joint_vel field above.
  - The flat max_joint_vel/min_joint_vel fields still apply to any phase NOT
    given an override in "phases" (fallback behavior), so only include phases
    that genuinely differ from the overall tendency.
"""

# Few-shot examples embedded in system prompt
_FEW_SHOT = """
Examples (description → JSON):

1. "快速抓起来" →
{
  "reasoning": "快速 → 高最小速度，无其他约束",
  "max_joint_vel": null,
  "min_joint_vel": 0.20,
  "kp_max": null,
  "kp_min": null,
  "grip_fraction_max": null,
  "grip_fraction_min": null,
  "max_jerk": null,
  "label": "快速抓取"
}

2. "像捧着豆腐一样轻轻拿起" →
{
  "reasoning": "轻轻+豆腐=极低速+极低刚度+轻握+平滑",
  "max_joint_vel": 0.04,
  "min_joint_vel": null,
  "kp_max": 380,
  "kp_min": null,
  "grip_fraction_max": 0.20,
  "grip_fraction_min": null,
  "max_jerk": 1.0,
  "label": "轻捧豆腐"
}

3. "把它用力压下去固定住" →
{
  "reasoning": "用力+固定=高刚度+大夹持力，速度中等",
  "max_joint_vel": 0.15,
  "min_joint_vel": null,
  "kp_max": null,
  "kp_min": 1200,
  "grip_fraction_max": null,
  "grip_fraction_min": 0.85,
  "max_jerk": null,
  "label": "用力按压"
}

4. "正常操作" →
{
  "reasoning": "无特殊要求",
  "max_joint_vel": null,
  "min_joint_vel": null,
  "kp_max": null,
  "kp_min": null,
  "grip_fraction_max": null,
  "grip_fraction_min": null,
  "max_jerk": null,
  "label": "正常操作"
}

5. "先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开" →
{
  "reasoning": "阶段变速：接近快/拿起轻/运送稳/放下慢/离开快",
  "max_joint_vel": null,
  "min_joint_vel": null,
  "kp_max": null,
  "kp_min": null,
  "grip_fraction_max": 0.30,
  "grip_fraction_min": null,
  "max_jerk": 1.0,
  "label": "阶段变速五段式",
  "phases": {
    "approach":  {"max_joint_vel": null, "min_joint_vel": 0.30},
    "descend":   {"max_joint_vel": 0.02, "min_joint_vel": null},
    "lift":      {"max_joint_vel": 0.04, "min_joint_vel": null},
    "transport": {"max_joint_vel": 0.04, "min_joint_vel": null},
    "final":     {"max_joint_vel": 0.015, "min_joint_vel": null},
    "retreat":   {"max_joint_vel": null, "min_joint_vel": 0.28}
  }
}
"""

_SYSTEM_PROMPT = textwrap.dedent(f"""
You are a robotic trajectory manner parser for a Franka Panda arm.

Your job: given a natural language description of HOW a manipulation task
should be performed (manner adverbials, metaphors, safety concerns, etc.),
output a JSON object with physical constraint parameters for the robot.

{_PARAM_GUIDE}

{_FEW_SHOT}

IMPORTANT:
- Output ONLY valid JSON, no markdown fences, no explanation outside the JSON.
- Set fields to null when there is no constraint in that dimension.
- "reasoning" field must be ≤ 30 Chinese characters.
- Be conservative: prefer slightly tighter constraints over loose ones for safety.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────────────────────────────────────

class LLMMannerParser:
    """Parse natural language manner descriptions into MannerSpec via Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai>=1.0")

        self.model = model or os.environ.get("MANNER_PARSER_MODEL", _DEFAULT_MODEL)
        self.temperature = temperature
        self.timeout = timeout

        _key = api_key or os.environ.get("AIHUBMIX_API_KEY", "")
        _url = base_url or os.environ.get("MANNER_PARSER_BASEURL", _DEFAULT_BASEURL)

        _proxy = os.environ.get("MANNER_PARSER_PROXY", "http://127.0.0.1:7897")
        _http_client = None
        if _proxy:
            import httpx
            _http_client = httpx.Client(proxy=_proxy, timeout=timeout)

        self._client = OpenAI(
            api_key=_key, base_url=_url, timeout=timeout,
            http_client=_http_client,
        )

    # ── public API ───────────────────────────────────────────────────────────

    def parse(self, description: str, *, tolerance: float = 0.15) -> MannerSpec:
        """Parse a natural language manner description into a MannerSpec.

        Parameters
        ----------
        description:
            任何语言描述如何执行操作，例如：
            "像怕吵醒婴儿一样轻轻放下玩具"
            "迅速抓起来扔进箱子"
            "小心翼翼地把玻璃杯立直"
        tolerance:
            Fractional overshoot tolerance passed to MannerSpec.

        Returns
        -------
        MannerSpec
            Physical constraints extracted from the description.
            Unconstrained dimensions are set to None.
        """
        raw_json = self._call_llm(description)
        spec = self._json_to_spec(raw_json, tolerance=tolerance)
        return spec

    def parse_with_reasoning(
        self, description: str, *, tolerance: float = 0.15
    ) -> tuple[MannerSpec, str]:
        """Like parse() but also return the LLM's reasoning string."""
        raw_json = self._call_llm(description)
        reasoning = raw_json.get("reasoning", "")
        spec = self._json_to_spec(raw_json, tolerance=tolerance)
        return spec, reasoning

    def batch_parse(
        self, descriptions: list[str], *, tolerance: float = 0.15
    ) -> list[MannerSpec]:
        """Parse multiple descriptions (sequential calls)."""
        return [self.parse(d, tolerance=tolerance) for d in descriptions]

    # ── private ──────────────────────────────────────────────────────────────

    def _call_llm(self, description: str) -> dict[str, Any]:
        """Call Gemini and return parsed JSON dict."""
        user_msg = f"描述: {description}"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        return self._extract_json(content)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences and truncation."""
        # strip ```json ... ``` fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        # find the opening brace
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response:\n{text[:300]}")
        candidate = text[start:]
        # try to parse as-is first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # try to find the last complete closing brace
        end = candidate.rfind("}")
        if end != -1:
            try:
                return json.loads(candidate[: end + 1])
            except json.JSONDecodeError:
                pass
        # last resort: patch truncated JSON by closing all open braces/brackets
        patched = candidate
        opens = patched.count("{") - patched.count("}")
        patched = patched.rstrip(",\n ") + "}" * max(opens, 1)
        try:
            return json.loads(patched)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"No JSON object found in LLM response:\n{text[:300]}"
            ) from exc

    @staticmethod
    def _json_to_spec(data: dict[str, Any], tolerance: float) -> MannerSpec:
        """Convert parsed JSON dict to MannerSpec."""
        def _f(key: str) -> float | None:
            v = data.get(key)
            if v is None or v == "null":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        label = str(data.get("label", ""))

        phase_velocity = None
        raw_phases = data.get("phases")
        if isinstance(raw_phases, dict) and raw_phases:
            _ALLOWED_PHASES = {
                "approach", "descend", "lift", "transport", "final", "retreat",
            }
            phase_velocity = {}
            for phase_name, bounds in raw_phases.items():
                if phase_name not in _ALLOWED_PHASES or not isinstance(bounds, dict):
                    continue

                def _pf(key: str) -> float | None:
                    v = bounds.get(key)
                    if v is None or v == "null":
                        return None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                mx, mn = _pf("max_joint_vel"), _pf("min_joint_vel")
                if mx is not None or mn is not None:
                    phase_velocity[phase_name] = PhaseVelocity(
                        max_joint_vel=mx, min_joint_vel=mn,
                    )
            if not phase_velocity:
                phase_velocity = None

        return MannerSpec(
            max_joint_vel=_f("max_joint_vel"),
            min_joint_vel=_f("min_joint_vel"),
            kp_max=_f("kp_max"),
            kp_min=_f("kp_min"),
            grip_fraction_max=_f("grip_fraction_max"),
            grip_fraction_min=_f("grip_fraction_min"),
            max_jerk=_f("max_jerk"),
            tolerance=tolerance,
            label=label,
            phase_velocity=phase_velocity,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience top-level function
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARSER: LLMMannerParser | None = None


def llm_parse_manner(
    description: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    tolerance: float = 0.15,
) -> MannerSpec:
    """Module-level convenience wrapper around LLMMannerParser.

    On first call, creates a singleton parser instance.

    Example::

        from capx.llm.manner_parser import llm_parse_manner
        spec = llm_parse_manner("像怕打碎玻璃一样轻轻放下方块")
        # → MannerSpec(max_joint_vel=0.04, kp_max=420, grip_fraction_max=0.22, max_jerk=0.8)
    """
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None or api_key or model:
        _DEFAULT_PARSER = LLMMannerParser(api_key=api_key, model=model)
    return _DEFAULT_PARSER.parse(description, tolerance=tolerance)
