"""Re-render demo videos with the current overlay (recovering scene frames from the old demo).

  python scripts/rerender.py outputs/vlm_chains/S4/ep00 [...]
"""
import json
import sys
from pathlib import Path

from dandori.render import recover_scene_frames, write_video

for d in map(Path, sys.argv[1:]):
    log = json.loads((d / "log.json").read_text())
    frames = recover_scene_frames(log, d / "demo.mp4")
    n_expected = log["attempts"][-1]["last_frame"] + 1
    print(d, "recovered", len(frames), "frames; log expects", n_expected)
    write_video(log, frames, d / "demo_v2.mp4")
