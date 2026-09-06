"""Print the default drawer stage plan."""

from __future__ import annotations

import json

from vlm_planner_cap.agents import VLMPlannerAgent, VLMVerifierAgent
from vlm_planner_cap.runner import PlannerVerifierLoop


def main() -> None:
    loop = PlannerVerifierLoop(
        planner=VLMPlannerAgent(),
        verifier=VLMVerifierAgent(),
    )
    preview = loop.dry_run_plan("open the top drawer and put the bowl inside")
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()

