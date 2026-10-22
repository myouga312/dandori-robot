"""Single-skill success rate of the executor (Go/No-go gate), using our own ChainSession.

Also checks that all tasks in the suite share one scene (prerequisite for goal swapping).

  python scripts/eval_skills.py --episodes 10 --out outputs/eval_skills
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio

from dandori.executor import DEFAULT_POLICY, Executor
from dandori.sim import ChainSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tasks", type=str, default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--videos", type=int, default=1, help="save video for the first N episodes per task")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_skills"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ex = Executor(args.policy, args.n_action_steps)
    probe = ChainSession(args.suite, 0, 0)
    instr = probe.instructions()
    for tid in range(len(instr)):  # raises if a task uses a different scene
        probe.set_goal(tid)
    probe.close()
    print(f"scene check OK: {len(instr)} tasks share one scene", flush=True)

    tids = [int(t) for t in args.tasks.split(",")] if args.tasks else list(range(len(instr)))
    rows = []
    for tid in tids:
        for ep in range(args.episodes):
            t0 = time.time()
            sess = ChainSession(args.suite, tid, ep)
            res = ex.run_skill(sess, instr[tid], max_steps=300, stop_on_success=True)
            sess.close()
            rows.append({"task_id": tid, "instruction": instr[tid], "episode": ep,
                         "success": res.gt_success, "steps": res.steps, "sec": round(time.time() - t0, 1)})
            print(json.dumps(rows[-1]), flush=True)
            if ep < args.videos:
                imageio.mimsave(args.out / f"task{tid:02d}_ep{ep:02d}.mp4", res.frames, fps=30)

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r["success"])
    summary = {
        "policy": args.policy, "n_action_steps": args.n_action_steps, "episodes": args.episodes,
        "per_task": {instr[t]: sum(v) / len(v) for t, v in by_task.items()},
        "overall": sum(r["success"] for r in rows) / len(rows),
    }
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
