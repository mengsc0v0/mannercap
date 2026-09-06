"""Initial agent interfaces for stage planning and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .pattern_priors import build_pattern_plan
from .schema import CoordinateObservation, StagePlan, StageStatus, TaskPlan


class ChatModel(Protocol):
    """Minimal chat model protocol used by planner/verifier agents."""

    def complete(self, messages: list[dict]) -> str:
        """Return model text for a chat-style message list."""


@dataclass
class VLMPlannerAgent:
    """Plan task stages from language, screenshots, and read-only coordinates."""

    model: ChatModel | None = None

    def plan(
        self,
        task: str,
        screenshot_path: str | None,
        coordinates: CoordinateObservation,
    ) -> TaskPlan:
        """Return a stage-level task plan.

        This implementation uses pattern priors distilled from the CAP-35
        success corpus. The priors are abstract stage schemas, not retrieved
        expert programs or task-specific few-shot code.
        """
        del screenshot_path, coordinates
        return build_pattern_plan(task)


@dataclass
class VLMVerifierAgent:
    """Verify whether a planned stage has been achieved."""

    model: ChatModel | None = None

    def verify(
        self,
        task: str,
        stage: StagePlan,
        screenshot_path: str | None,
        coordinates: CoordinateObservation,
        stdout: str = "",
        stderr: str = "",
    ) -> tuple[StageStatus, str]:
        """Return a stage decision and short rationale.

        This skeleton uses conservative rule-based checks until a model-backed
        verifier prompt is wired in.
        """
        del task, screenshot_path, coordinates, stdout
        if stderr.strip():
            return StageStatus.NEEDS_REPAIR, "Execution produced stderr; repair the stage code first."
        return (
            StageStatus.UNCERTAIN,
            f"No verifier implementation is wired for stage '{stage.stage_id}' yet.",
        )

