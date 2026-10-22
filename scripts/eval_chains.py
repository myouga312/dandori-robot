"""Multi-step (no reset) evaluation.

  # executor only: given plan, oracle stop (no API key needed)
  python scripts/eval_chains.py --mode oracle --given-plan --episodes 10
  # full system: planner + VLM stop / retry / replan
  python scripts/eval_chains.py --mode vlm --episodes 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from dandori.executor import DEFAULT_POLICY, Executor
from dandori.orchestrator import Config, run_request
from dandori.render import write_video
from dandori.sim import ChainSession
from dandori.skills import LIBERO_GOAL

SKILL_IDS = [s.instruction for s in LIBERO_GOAL]  # same order as the LIBERO suite (checked in eval_skills)

SCENARIOS = {
    "S1": ("料理の準備をして。コンロをつけて、ボウルをその上に載せておいて。",
           ["turn on the stove", "put the bowl on the stove"]),
    "S2": ("おつまみの用意をして。クリームチーズをボウルに入れて、お皿に載せてね。",
           ["put the cream cheese in the bowl", "put the bowl on the plate"]),
    "S3": ("引き出しを開けて、ワインは棚の上へ。最後にコンロもつけておいて。",
           ["open the middle drawer of the cabinet", "put the wine bottle on top of the cabinet", "turn on the stove"]),
    "S4": ("ワインは棚の上に片付けて、コンロをつけたら、ボウルはお皿に置いておいて。",
           ["put the wine bottle on top of the cabinet", "turn on the stove", "put the bowl on the plate"]),
    "S5": ("ワインを棚の上に置いてから、真ん中の引き出しを開けて。",
           ["put the wine bottle on top of the cabinet", "open the middle drawer of the cabinet"]),
}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "vlm"], default="vlm")
    ap.add_argument("--given-plan", action="store_true", help="skip the planner, use the scenario's plan")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=1)
    ap.add_argument("--check-every", type=int, default=100)
    ap.add_argument("--no-home", action="store_true", help="disable return-to-start between skills")
    ap.add_argument("--videos", type=int, default=2, help="render the first N episodes per scenario")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_chains"))
    args = ap.parse_args()

    ex = Executor(args.policy)
    rows = []
    tag = f"{args.mode}{'_given' if args.given_plan else ''}{'_nohome' if args.no_home else ''}"
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = (args.out / f"rows_{tag}.jsonl").open("a")
    for sid in args.scenarios.split(","):
        request, plan = SCENARIOS[sid]
        for ep in range(args.start_episode, args.start_episode + args.episodes):
            cfg = Config(mode=args.mode, check_every=args.check_every, max_retries=args.max_retries,
                         max_replans=args.max_replans, home_between=not args.no_home, gt_plan=plan if args.given_plan else None)
            # start from the benchmark init state of the reference plan's first task
            sess = ChainSession("libero_goal", init_task_id=SKILL_IDS.index(plan[0]), episode_index=ep)
            out = args.out / sid / f"ep{ep:02d}"
            log, frames = run_request(request, ex, sess, cfg, out)
            # ground truth against the scenario's reference goals (independent of what the planner chose)
            ref_gt = {}
            for g in plan:
                sess.set_goal(sess.task_id_of(g))
                ref_gt[g] = sess.is_success()
            sess.close()
            row = {"scenario": sid, "episode": ep, "ref_done_gt": all(ref_gt.values()), "ref_gt": ref_gt,
                   "all_done_gt": log.all_done_gt,
                   "plan": log.plans[0]["steps"], "plan_matches_ref": log.plans[0]["steps"] == plan,
                   "goals_gt": log.goals_gt, "n_attempts": len(log.attempts),
                   "vlm_calls": sum(a["vlm_calls"] for a in log.attempts), "wall_sec": round(log.wall_sec, 1)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            if ep - args.start_episode < args.videos:
                write_video(asdict(log), frames, out / "demo.mp4")
    (args.out / f"rows_{tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
