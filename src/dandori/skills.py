"""Skill library: the language instructions the VLA was trained on.

The planner may only emit skills from this list, so every planned step is
something the executor policy has actually seen during training.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    instruction: str  # exact training instruction passed to the VLA
    ja: str           # Japanese label for subtitles / logs


# LIBERO-Goal: one kitchen scene, 10 goal tasks.
# Verified against the LIBERO benchmark task list at runtime (see env.py).
LIBERO_GOAL = [
    Skill("open the middle drawer of the cabinet", "棚の真ん中の引き出しを開ける"),
    Skill("put the bowl on the stove", "ボウルをコンロに置く"),
    Skill("put the wine bottle on top of the cabinet", "ワインボトルを棚の上に置く"),
    Skill("open the top drawer and put the bowl inside", "上の引き出しを開けてボウルを入れる"),
    Skill("put the bowl on top of the cabinet", "ボウルを棚の上に置く"),
    Skill("push the plate to the front of the stove", "皿をコンロの手前に押す"),
    Skill("put the cream cheese in the bowl", "クリームチーズをボウルに入れる"),
    Skill("turn on the stove", "コンロをつける"),
    Skill("put the bowl on the plate", "ボウルを皿に置く"),
    Skill("put the wine bottle on the rack", "ワインボトルをラックに置く"),
]

SUITES = {"libero_goal": LIBERO_GOAL}


def by_instruction(suite: str) -> dict[str, Skill]:
    return {s.instruction: s for s in SUITES[suite]}
