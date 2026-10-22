"""Assemble the submission video (1280x720, 30 fps, about 2.5 min).

  python scripts/make_final_video.py --out outputs/final/PAI_final.mp4
Uses: demo_v2.mp4 of selected episodes, results/figures, results/planner_examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from dandori.render import ACCENT, BG, FG, H, MUTED, NG, OK, W, WARN, _font, _wrap

FPS = 30


def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def text_block(d, x, y, text, font, fill=FG, width=W - 160, gap=10) -> int:
    for line in _wrap(d, text, font, width):
        d.text((x, y), line, font=font, fill=fill)
        y += font.size + gap
    return y


def title_card() -> Image.Image:
    img, d = card()
    d.text((80, 200), "段取りロボット", font=_font(72), fill=FG)
    d.text((84, 300), "一つの動作しか知らないVLAに、日本語の「段取り」をこなさせる", font=_font(30), fill=ACCENT)
    d.text((84, 380), "LLM Planner → SmolVLA → VLM Verifier ／ LIBERO-Goal シミュレーション", font=_font(24), fill=MUTED)
    d.text((84, 600), "Physical AI 応用編1 最終課題", font=_font(22), fill=MUTED)
    return img


def problem_card() -> Image.Image:
    img, d = card()
    d.text((80, 70), "課題", font=_font(26), fill=ACCENT)
    y = text_block(d, 80, 120, "VLA（SmolVLA）は「コンロをつける」のような1つの指示なら高い成功率で実行できる（単体 77%）。"
                   "しかし「料理の準備をして」のような、複数の手順が要るあいまいな指示には対応できない。", _font(32))
    y = text_block(d, 80, y + 30, "単純に手順をつなぐだけでは、2手順目以降の成功率は 9%（6/70）まで落ちる。", _font(32), fill=NG)
    d.text((80, y + 50), "原因：前の手順を終えたアームの姿勢が、学習時の開始姿勢と違う（分布のずれ）", font=_font(26), fill=MUTED)
    return img


def arch_card() -> Image.Image:
    img, d = card()
    d.text((80, 50), "システム構成", font=_font(26), fill=ACCENT)
    boxes = [("LLM Planner", "日本語の指示 → スキル一覧の中から手順の列\n（Claude、JSON Schemaで出力を制約）", 120),
             ("SmolVLA Executor", "1手順ずつ閉ループで実行。リセットせずシーンを引き継ぐ\n手順の間に関節空間のホーム復帰", 300),
             ("VLM Verifier", "100ステップごとに実行前後の画像で達成を判定\n→ 停止 ／ 再試行 ／ 再計画", 480)]
    for name, desc, y in boxes:
        d.rounded_rectangle((80, y, 420, y + 130), radius=14, outline=ACCENT, width=3)
        d.text((104, y + 46), name, font=_font(30), fill=FG)
        for i, line in enumerate(desc.split("\n")):
            d.text((460, y + 30 + i * 40), line, font=_font(25), fill=FG)
    for y in (250, 430):
        d.polygon([(240, y), (260, y), (250, y + 50)], fill=MUTED)
        d.rectangle((247, y - 0, 253, y + 30), fill=MUTED)
    d.text((80, 650), "LLM/VLMの呼び出しは、GPU VM から GCS を介して Claude Code（非対話モード）へ中継", font=_font(20), fill=MUTED)
    return img


def section_card(title: str, sub: str) -> Image.Image:
    img, d = card()
    d.text((80, 280), title, font=_font(52), fill=FG)
    text_block(d, 84, 370, sub, _font(28), fill=MUTED)
    return img


def planner_card(examples: dict) -> Image.Image:
    img, d = card()
    d.text((80, 50), "実行できない依頼は、計画の段階で断る（Planner の実際の出力）", font=_font(26), fill=ACCENT)
    y = 110
    for req, p in examples.items():
        d.text((80, y), f"「{req}」", font=_font(30), fill=FG)
        y += 50
        steps = "、".join(p["steps"]) if p["steps"] else "（なし）"
        d.text((110, y), f"計画: {steps}", font=_font(22), fill=OK if p["steps"] else MUTED)
        y += 36
        y = text_block(d, 110, y, p["ja_summary"], _font(24), fill=WARN, width=W - 220, gap=8) + 40
    return img


def results_card(fig: Path, sys_rows: list[dict]) -> Image.Image:
    img, d = card()
    d.text((60, 30), "評価（LIBERO-Goal、各シナリオ10エピソード、シミュレータの正解判定）", font=_font(24), fill=ACCENT)
    f = Image.open(fig).convert("RGB")
    f.thumbnail((700, 600))
    img.paste(f, (40, 90))
    x = 780
    d.text((x, 100), "全目標の達成（5シナリオ計）", font=_font(24), fill=FG)
    rows = [("単純につなぐ", "1/50", MUTED), ("＋ホーム復帰", "14/50", FG), ("＋Verifier（停止・再試行・再計画）", f"{sum(r['ref_done_gt'] for r in sys_rows)}/50", OK)]
    y = 150
    for name, v, col in rows:
        d.text((x, y), name, font=_font(22), fill=MUTED)
        d.text((x + 10, y + 30), v, font=_font(44), fill=col)
        y += 110
    d.text((x, y + 10), "Verifier の判定正解率 92.7%", font=_font(22), fill=FG)
    d.text((x, y + 44), "Planner の計画の正しさ 50/50", font=_font(22), fill=FG)
    return img


def summary_card() -> Image.Image:
    img, d = card()
    d.text((80, 80), "まとめ", font=_font(30), fill=ACCENT)
    y = 150
    for line in ["単体スキル用のVLAを、追加学習なしで複数の手順に使えるようにした",
                 "成功率を最も押し上げたのは、手順の間の「関節空間ホーム復帰」（9% → 61%）",
                 "VLM Verifier による停止・再試行・再計画で、全目標の達成が 14/50 → 23/50",
                 "残る失敗：前の手順が次の前提を変える組み合わせ（例：中身入りのボウル）",
                 "今後：手順の相性を考える計画、失敗からの回復を学習、実機（SO-101）への展開"]:
        y = text_block(d, 100, y, "・" + line, _font(28), gap=8) + 24
    return img


def hold(w, img: Image.Image, sec: float) -> None:
    a = np.asarray(img)
    for _ in range(int(sec * FPS)):
        w.append_data(a)


def clip(w, path: Path, speed: int = 1, banner: str | None = None) -> None:
    font = _font(26)
    for i, fr in enumerate(imageio.get_reader(path)):
        if i % speed:
            continue
        if banner:
            im = Image.fromarray(fr)
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((W - 170, H - 56, W - 16, H - 14), radius=10, fill=(0, 0, 0))
            d.text((W - 154, H - 50), banner, font=font, fill=WARN)
            fr = np.asarray(im)
        w.append_data(fr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("outputs/vlm_chains"))
    ap.add_argument("--out", type=Path, default=Path("outputs/final/PAI_final.mp4"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples = json.loads(Path("results/planner_examples/examples.json").read_text())
    sys_rows = json.loads(Path("results/vlm_chains.json").read_text())
    with imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as w:
        hold(w, title_card(), 5)
        hold(w, problem_card(), 9)
        hold(w, arch_card(), 11)
        hold(w, section_card("デモ1　2手順", "「料理の準備をして。コンロをつけて、ボウルをその上に載せておいて。」"), 3)
        clip(w, args.runs / "S1/ep00/demo_v2.mp4")
        hold(w, section_card("デモ2　3手順", "「ワインは棚の上に片付けて、コンロをつけたら、ボウルはお皿に置いておいて。」"), 3)
        clip(w, args.runs / "S4/ep01/demo_v2.mp4")
        hold(w, section_card("デモ3　失敗からの立て直し（2倍速）",
                             "ボウルの置き方に失敗 → Verifierが検知 → 再試行 → 再計画 → 成功"), 4)
        clip(w, args.runs / "S4/ep00/demo_v2.mp4", speed=2, banner="2倍速")
        hold(w, planner_card(examples), 12)
        hold(w, results_card(Path("results/figures/fig_home_ablation.png"), sys_rows), 14)
        hold(w, summary_card(), 10)
    print("saved", args.out)


if __name__ == "__main__":
    main()
