"""Compose demo videos: scene on the left, request / plan / verdicts on the right."""
from __future__ import annotations

import glob
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .skills import by_instruction

W, H = 1280, 720
SCENE = 720
BG = (18, 22, 30)
FG = (235, 238, 242)
MUTED = (140, 148, 160)
ACCENT = (90, 170, 255)
OK = (80, 200, 120)
NG = (240, 90, 90)
WARN = (255, 190, 80)

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES + glob.glob("/usr/share/fonts/**/*CJK*", recursive=True):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("No CJK font found (apt install fonts-noto-cjk)")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


class Renderer:
    def __init__(self, log: dict, title: str = "段取りロボット: LLM Planner → VLA → VLM Verifier"):
        self.log = log
        self.title = title
        self.lib = by_instruction(log["suite"])
        self.f_title, self.f_big, self.f_mid, self.f_small = _font(22), _font(30), _font(24), _font(19)

    def _status_at(self, f: int) -> tuple[list[tuple[str, str]], dict | None]:
        """Plan rows [(instruction, state)] and the attempt active at frame f."""
        plan = list(self.log["plans"][0]["steps"])
        states = {s: "todo" for s in plan}
        active = None
        n_plans = 1
        failed_before: set[str] = set()
        for a in self.log["attempts"]:
            if a["first_frame"] > f:
                break
            if a["instruction"] not in states:  # added by a replan
                plan.append(a["instruction"])
            if a["last_frame"] >= f:
                if a["try_index"] > 0:
                    states[a["instruction"]] = "retry"
                elif a["instruction"] in failed_before:  # same step again after the planner re-planned
                    states[a["instruction"]] = "replan"
                else:
                    states[a["instruction"]] = "run"
                active = a
            else:
                ok = a["vlm_success"] if a["vlm_success"] is not None else a["gt_success"]
                states[a["instruction"]] = "done" if ok else "fail"
                if not ok:
                    failed_before.add(a["instruction"])
        return [(s, states.get(s, "todo")) for s in plan], active

    def frame(self, scene: np.ndarray, f: int, banner: str | None = None) -> np.ndarray:
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(scene).resize((SCENE, SCENE), Image.BILINEAR), (0, 0))
        d = ImageDraw.Draw(img)
        x, y, w = SCENE + 28, 24, W - SCENE - 52
        d.text((x, y), self.title, font=self.f_small, fill=MUTED)
        y += 40
        d.text((x, y), "指示", font=self.f_small, fill=ACCENT)
        y += 28
        for line in _wrap(d, f"「{self.log['request_ja']}」", self.f_big, w):
            d.text((x, y), line, font=self.f_big, fill=FG)
            y += 40
        y += 16
        d.text((x, y), "計画（LLM Planner）", font=self.f_small, fill=ACCENT)
        y += 30
        rows, active = self._status_at(f)
        # glyphs limited to what Noto Sans CJK / Hiragino both have
        icon = {"todo": ("・", MUTED, ""), "run": ("▶", ACCENT, ""), "retry": ("▶", WARN, "（再試行）"),
                "replan": ("▶", WARN, "（再計画）"), "done": ("✓", OK, ""), "fail": ("×", NG, "")}
        for i, (instr, st) in enumerate(rows, 1):
            mark, col, note = icon[st]
            ja = self.lib[instr].ja if instr in self.lib else instr
            d.text((x, y), f"{mark} {i}. {ja}{note}", font=self.f_mid, fill=col if st != "todo" else FG)
            y += 32
            d.text((x + 34, y), instr, font=self.f_small, fill=MUTED)
            y += 30
        y += 12
        if active is not None:
            step = f - active["first_frame"] + 1
            d.text((x, y), f"実行中（SmolVLA）  step {step}", font=self.f_small, fill=ACCENT)
            y += 30
        last = [a for a in self.log["attempts"] if a["last_frame"] < f and a["vlm_reason"]]
        if last:
            a = last[-1]
            col = OK if a["vlm_success"] else NG
            d.text((x, y), "判定（VLM Verifier）: " + ("成功" if a["vlm_success"] else "失敗"), font=self.f_small, fill=col)
            y += 28
            for line in _wrap(d, a["vlm_reason"], self.f_small, w)[:4]:
                d.text((x, y), line, font=self.f_small, fill=FG)
                y += 26
        if banner:
            d.rectangle((0, H - 64, SCENE, H), fill=(0, 0, 0))
            d.text((20, H - 50), banner, font=self.f_mid, fill=FG)
        return np.asarray(img)


def write_video(log: dict, frames: list[np.ndarray], path: Path, fps: int = 30, hold_sec: float = 1.5,
                save_scene: bool = True) -> None:
    r = Renderer(log)
    hold = int(fps * hold_sec)
    if save_scene:  # raw scene frames, so the overlay can be re-rendered later
        imageio.mimsave(path.with_name("scene.mp4"), frames, fps=fps, quality=9, macro_block_size=8)
    ends = {a["last_frame"]: a for a in log["attempts"]}
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for _ in range(hold * 2):
            w.append_data(r.frame(frames[0], 0, "指示を受け取り、計画を立てました"))
        for f, scene in enumerate(frames):
            w.append_data(r.frame(scene, f))
            if f in ends:  # pause on each verdict
                for _ in range(hold):
                    w.append_data(r.frame(scene, f + 1))
        result = "すべての目標を達成" if log["all_done_gt"] else "未達成の目標あり"
        for _ in range(hold * 2):
            w.append_data(r.frame(frames[-1], len(frames), f"結果（シミュレータ判定）: {result}"))


def recover_scene_frames(log: dict, demo_mp4: Path, fps: int = 30, hold_sec: float = 1.5) -> list[np.ndarray]:
    """Invert write_video(): recover the raw scene frames from an already-rendered demo."""
    hold = int(fps * hold_sec)
    ends = {a["last_frame"] for a in log["attempts"]}
    video = [fr for fr in imageio.get_reader(demo_mp4)]
    body = video[hold * 2: len(video) - hold * 2]
    frames, i, f = [], 0, 0
    while i < len(body):
        frames.append(np.asarray(Image.fromarray(body[i][:SCENE, :SCENE]).resize((360, 360), Image.BILINEAR)))
        i += 1 + (hold if f in ends else 0)
        f += 1
    return frames
