"""Verifier: judge from camera images whether a skill has been accomplished."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import llm

SYSTEM = """\
You verify whether a robot arm has completed a manipulation instruction in a kitchen
simulation. You receive the scene before and after the attempt, from a fixed third-person
camera (and optionally a wrist camera). Judge only the final state in the AFTER image
against the instruction's goal condition. Be strict: an object that is lifted but not
resting at the target, or a drawer only slightly opened, counts as not done.
Write `reason_ja` in Japanese, one or two sentences.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason_ja": {"type": "string"},
    },
    "required": ["success", "confidence", "reason_ja"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    success: bool
    confidence: float
    reason_ja: str
    meta: dict = field(default_factory=dict)


def verify(
    instruction: str,
    before: np.ndarray,
    after: np.ndarray,
    after_wrist: np.ndarray | None = None,
    model: str | None = None,
    gt: bool | None = None,
) -> Verdict:
    """gt: simulator ground truth at call time; only used to build an offline verifier dataset
    when VERIFY_DUMP_DIR is set (never shown to the model)."""
    content = [
        {"type": "text", "text": "BEFORE (third-person camera):"},
        llm.image_block(before),
        {"type": "text", "text": "AFTER (third-person camera):"},
        llm.image_block(after),
    ]
    if after_wrist is not None:
        content += [{"type": "text", "text": "AFTER (wrist camera):"}, llm.image_block(after_wrist)]
    content.append({"type": "text", "text": f"Instruction: {instruction}\nWas it completed?"})
    out, meta = llm.call_json(SYSTEM, content, SCHEMA, model=model or llm.model_from_env("VERIFIER_MODEL"))
    v = Verdict(bool(out["success"]), float(out["confidence"]), out["reason_ja"], meta)
    if os.environ.get("VERIFY_DUMP_DIR"):
        _dump(Path(os.environ["VERIFY_DUMP_DIR"]), instruction, before, after, after_wrist, v, gt)
    return v


def _dump(root: Path, instruction, before, after, after_wrist, v: Verdict, gt) -> None:
    from PIL import Image

    d = root / f"{time.time_ns()}"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(before).save(d / "before.png")
    Image.fromarray(after).save(d / "after.png")
    if after_wrist is not None:
        Image.fromarray(after_wrist).save(d / "wrist.png")
    (d / "meta.json").write_text(json.dumps(
        {"instruction": instruction, "gt": gt, "verdict": v.success, "confidence": v.confidence,
         "reason_ja": v.reason_ja, "llm": v.meta}, ensure_ascii=False))
