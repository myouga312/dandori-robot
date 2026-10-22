"""Figures for the report.

  python scripts/plot_results.py --chains-home outputs/oracle_chains/home --chains-nohome outputs/oracle_chains/nohome
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from dandori.skills import by_instruction

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2 (validated: scripts in dataviz skill)

for f in ["Hiragino Sans", "Noto Sans CJK JP", "Noto Sans CJK"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break


def step_success(run_dir: Path) -> dict[tuple[str, int], tuple[int, int, str]]:
    """(scenario, step index) -> (successes, trials, instruction) at the end of each step."""
    agg: dict = collections.defaultdict(lambda: [0, 0, ""])
    for f in glob.glob(str(run_dir / "S*/ep*/log.json")):
        log = json.loads(Path(f).read_text())
        sid = Path(f).parts[-3]
        seen = set()
        for a in log["attempts"]:  # count each planned step once (last try wins)
            seen.add(a["instruction"])
        idx = {s: i for i, s in enumerate(log["plans"][0]["steps"])}
        last = {}
        for a in log["attempts"]:
            last[a["instruction"]] = a["gt_success"]
        for instr, ok in last.items():
            k = (sid, idx.get(instr, -1))
            agg[k][0] += ok
            agg[k][1] += 1
            agg[k][2] = instr
    return {k: tuple(v) for k, v in agg.items()}


def fig_home(home: Path, nohome: Path, out: Path) -> None:
    h, n = step_success(home), step_success(nohome)
    ja = by_instruction("libero_goal")
    keys = sorted(k for k in h if k[1] >= 1)  # steps after the first: where the start pose matters
    labels = [f"{k[0]}-{k[1] + 1}  {ja[h[k][2]].ja}" for k in keys]
    vh = [h[k][0] / h[k][1] * 100 for k in keys]
    vn = [n[k][0] / n[k][1] * 100 if k in n else 0 for k in keys]

    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(keys) + 1.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bh = 0.36
    ys = range(len(keys))
    for i, (vals, name, col, off) in enumerate([(vh, "ホーム復帰あり", SERIES[0], -bh / 2 - 0.01),
                                                (vn, "ホーム復帰なし", SERIES[1], bh / 2 + 0.01)]):
        bars = ax.barh([y + off for y in ys], vals, height=bh, color=col, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8, color=TEXT2)
    ax.set_yticks(list(ys), labels, fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("その手順を終えた時点の成功率（%、各10エピソード）", fontsize=9, color=TEXT2)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    tot_h = sum(h[k][0] for k in keys), sum(h[k][1] for k in keys)
    tot_n = sum(n[k][0] for k in keys if k in n), sum(n[k][1] for k in keys if k in n)
    ax.set_title(f"2手順目以降の成功率：ホーム復帰なし {tot_n[0]}/{tot_n[1]} → あり {tot_h[0]}/{tot_h[1]}",
                 fontsize=11, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-home", type=Path, required=True)
    ap.add_argument("--chains-nohome", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_home(args.chains_home, args.chains_nohome, args.out / "fig_home_ablation.png")


if __name__ == "__main__":
    main()
