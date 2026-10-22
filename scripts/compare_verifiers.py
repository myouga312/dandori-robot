"""Offline verifier comparison on dumped verifier calls (images + simulator ground truth).

  LLM_BACKEND=codex-cli python scripts/compare_verifiers.py outputs/vlm_chains/verifier_dump
Re-judges every sample with the current backend and reports both the original (online)
verdicts and the new ones against ground truth.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from dandori import llm, verifier


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(p and g for p, g in pairs); tn = sum((not p) and (not g) for p, g in pairs)
    fp = sum(p and not g for p, g in pairs); fn = sum((not p) and g for p, g in pairs)
    n = len(pairs)
    return {"n": n, "acc": round((tp + tn) / n, 3) if n else None, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    samples = sorted(p for p in args.dump.iterdir() if (p / "meta.json").exists())
    if args.limit:
        samples = samples[: args.limit]
    out_path = args.dump.parent / f"verifier_compare_{llm.backend()}.jsonl"
    done = {json.loads(l)["id"] for l in out_path.open()} if out_path.exists() else set()
    lock = threading.Lock()

    def judge(d: Path, f) -> None:
        m = json.loads((d / "meta.json").read_text())
        img = lambda n: np.asarray(Image.open(d / n).convert("RGB")) if (d / n).exists() else None
        try:
            v = verifier.verify(m["instruction"], img("before.png"), img("after.png"), img("wrist.png"))
            row = {"id": d.name, "instruction": m["instruction"], "gt": m["gt"], "online": m["verdict"],
                   "new": v.success, "reason_ja": v.reason_ja, "sec": v.meta.get("sec")}
        except Exception as e:  # keep going; record the failure
            row = {"id": d.name, "error": str(e)[:300]}
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        for d in samples:
            if d.name not in done:
                pool.submit(judge, d, f)
    rows = [json.loads(l) for l in out_path.open() if "error" not in l]
    print(json.dumps({"online (as run)": confusion([(r["online"], r["gt"]) for r in rows]),
                      llm.backend(): confusion([(r["new"], r["gt"]) for r in rows])}, indent=1))


if __name__ == "__main__":
    main()
