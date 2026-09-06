"""Skeleton runner for the planner-verifier CaP loop."""

from __future__ import annotations

from dataclasses import dataclass

from .agents import VLMPlannerAgent, VLMVerifierAgent
from .schema import CoordinateObservation, StageStatus


@dataclass
class PlannerVerifierLoop:
    """Coordinate planner, executor, and verifier.

    The executor is intentionally not implemented here yet. It will be the
    bridge to the copied CaP-X environment runner or a clean LIBERO adapter.
    """

    planner: VLMPlannerAgent
    verifier: VLMVerifierAgent

    def dry_run_plan(
        self,
        task: str,
        screenshot_path: str | None = None,
        coordinates: CoordinateObservation | None = None,
    ) -> list[dict]:
        """Return a serializable preview of the stage plan."""
        coordinates = coordinates or CoordinateObservation()
        plan = self.planner.plan(task, screenshot_path, coordinates)
        preview = []
        for stage in plan.stages:
            status, rationale = self.verifier.verify(
                task, stage, screenshot_path, coordinates,
            )
            preview.append(
                {
                    "stage_id": stage.stage_id,
                    "name": stage.name,
                    "intent": stage.intent,
                    "family": stage.family.value if stage.family is not None else None,
                    "max_attempts": stage.max_attempts,
                    "allowed_primitives": stage.allowed_primitives,
                    "allowed_observations": stage.allowed_observations,
                    "checks": [check.name for check in stage.checks],
                    "initial_status": status.value
                    if isinstance(status, StageStatus)
                    else str(status),
                    "initial_rationale": rationale,
                }
            )
        return preview
