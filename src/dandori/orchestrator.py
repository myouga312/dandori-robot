"""Plan -> execute -> verify -> retry / replan loop for one Japanese request."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import planner, verifier
from .executor import Executor
from .sim import ChainSession
from .skills import SUITES, by_instruction


@dataclass
class Attempt:
    instruction: str
    try_index: int
    steps: int
    gt_success: bool
    vlm_success: bool | None
    vlm_confidence: float | None
    vlm_reason: str | None
    first_frame: int  # index into the episode frame list
    last_frame: int
    vlm_calls: int


@dataclass
class EpisodeLog:
    request_ja: str
    suite: str
    init_task_id: int
    episode_index: int
    mode: str
    plans: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    goals_gt: dict = field(default_factory=dict)  # instruction -> satisfied at the end
    all_done_gt: bool = False
    wall_sec: float = 0.0


@dataclass
class Config:
    # "oracle": stop each skill on the simulator predicate, no verifier (upper bound, not deployable)
    # "vlm":    VLM checks every `check_every` steps to stop; final verdict drives retry / replan
    mode: str = "vlm"
    check_every: int = 100
    max_retries: int = 1          # extra tries of the same skill after a failed verdict
    max_replans: int = 1
    max_steps_per_skill: int = 300
    home_between: bool = True     # scripted return to the start pose before each skill (except the first)
    grace_steps: int = 25         # let the policy finish its motion (release) after a stop decision
    gt_plan: list[str] | None = None  # bypass the planner (to isolate executor effects)


def run_request(
    request_ja: str,
    executor: Executor,
    sess: ChainSession,
    cfg: Config,
    out_dir: Path | None = None,
) -> tuple[EpisodeLog, list[np.ndarray]]:
    t0 = time.time()
    skills = SUITES[sess.suite_name]
    lib = by_instruction(sess.suite_name)
    log = EpisodeLog(request_ja, sess.suite_name, sess.task_id, sess.env.episode_index, cfg.mode)
    frames: list[np.ndarray] = [sess.frame()]

    if cfg.gt_plan is not None:
        queue = list(cfg.gt_plan)
        log.plans.append({"steps": list(queue), "source": "given"})
    else:
        p = planner.plan(request_ja, skills, scene=sess.frame())
        queue = list(p.steps)
        log.plans.append({"steps": p.steps, "feasible": p.feasible, "reason": p.reason,
                          "ja_summary": p.ja_summary, "meta": p.meta})
    planned_goals = list(queue)

    done: list[str] = []
    replans = 0
    while queue:
        instr = queue.pop(0)
        for k in range(1 + cfg.max_retries):
            if cfg.home_between and log.attempts:
                sess.go_home(frames=frames)
            sess.set_goal(sess.task_id_of(instr))
            before = sess.frame()
            start = len(frames)
            verdicts: list[verifier.Verdict] = []

            def periodic_check(step: int) -> bool:
                if cfg.mode != "vlm" or step % cfg.check_every:
                    return False
                verdicts.append(verifier.verify(instr, before, sess.frame(), sess.wrist(), gt=sess.is_success()))
                return verdicts[-1].success

            res = executor.run_skill(sess, instr, cfg.max_steps_per_skill,
                                     stop_on_success=cfg.mode == "oracle", should_stop=periodic_check,
                                     grace_steps=cfg.grace_steps)
            frames += res.frames
            v = None
            if cfg.mode == "vlm":
                # reuse the stopping verdict; otherwise judge the final state once
                v = verdicts[-1] if res.early_stop and verdicts else verifier.verify(instr, before, sess.frame(), sess.wrist(), gt=sess.is_success())
            log.attempts.append(asdict(Attempt(
                instr, k, res.steps, sess.is_success(),
                v.success if v else None, v.confidence if v else None, v.reason_ja if v else None,
                start, len(frames) - 1, len(verdicts))))
            if v is None or v.success:
                break
        if v is not None and not v.success:
            if replans >= cfg.max_replans:
                break
            replans += 1
            p = planner.plan(request_ja, skills, scene=sess.frame(), done=done, failed_note=v.reason_ja)
            queue = list(p.steps)
            log.plans.append({"steps": p.steps, "feasible": p.feasible, "reason": p.reason,
                              "ja_summary": p.ja_summary, "meta": p.meta, "replan": True})
            planned_goals += [s for s in p.steps if s not in planned_goals]
            continue
        done.append(instr)

    # Ground truth: are all planned goals satisfied in the final scene?
    for g in planned_goals:
        sess.set_goal(sess.task_id_of(g))
        log.goals_gt[g] = sess.is_success()
    log.all_done_gt = bool(planned_goals) and all(log.goals_gt.values())
    log.wall_sec = time.time() - t0

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "log.json").write_text(json.dumps(asdict(log), ensure_ascii=False, indent=2))
    return log, frames
