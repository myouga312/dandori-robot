"""Planner: Japanese instruction (+ current scene image) -> sequence of library skills."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import llm
from .skills import Skill

SYSTEM = """\
You are the task planner for a single-arm kitchen robot.
The robot can only execute the skills listed below, each exactly as written.
Given a user's request in Japanese and an image of the current scene, output the
shortest ordered list of skills that satisfies the request.

Rules:
- Use only skills from the list. Never invent new ones.
- Respect physical preconditions (e.g. an object already placed somewhere cannot be
  placed elsewhere unless it is picked again by another listed skill; a skill whose goal
  is already satisfied in the image should be skipped).
- If the request cannot be satisfied with the listed skills, set feasible=false,
  return the best partial plan, and explain what is missing.
- Write `reason` and `ja_summary` in Japanese.
"""


@dataclass
class Plan:
    steps: list[str]
    feasible: bool
    reason: str
    ja_summary: str
    meta: dict = field(default_factory=dict)


def _schema(skills: list[Skill]) -> dict:
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string", "enum": [s.instruction for s in skills]},
            },
            "feasible": {"type": "boolean"},
            "reason": {"type": "string"},
            "ja_summary": {"type": "string"},
        },
        "required": ["steps", "feasible", "reason", "ja_summary"],
        "additionalProperties": False,
    }


def plan(
    request_ja: str,
    skills: list[Skill],
    scene: np.ndarray | None = None,
    done: list[str] | None = None,
    failed_note: str | None = None,
    model: str | None = None,
) -> Plan:
    """Create (or re-create) a plan.

    done:        skills already completed in this episode (for replanning)
    failed_note: verifier's explanation of the last failure (for replanning)
    """
    lib = "\n".join(f"- {s.instruction}  ({s.ja})" for s in skills)
    text = f"Skill library:\n{lib}\n\nUser request (Japanese): {request_ja}\n"
    if done:
        text += "\nAlready completed: " + "; ".join(done) + "\n"
    if failed_note:
        text += f"\nThe last attempt failed. Verifier note: {failed_note}\nPlan the remaining steps.\n"
    content: list[dict] = []
    if scene is not None:
        content.append(llm.image_block(scene))
    content.append({"type": "text", "text": text})
    out, meta = llm.call_json(SYSTEM, content, _schema(skills), model=model or llm.model_from_env("PLANNER_MODEL"))
    return Plan(out["steps"], out["feasible"], out["reason"], out["ja_summary"], meta)
