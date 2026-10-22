"""Executor: run one language-conditioned skill with a pretrained LeRobot VLA."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors

from .sim import OBS_SIZE, ChainSession

DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"


@dataclass
class SkillResult:
    instruction: str
    steps: int
    gt_success: bool          # simulator predicate for this skill's goal
    early_stop: bool          # stopped because the predicate fired
    frames: list[np.ndarray] = field(default_factory=list)


class Executor:
    def __init__(self, policy_path: str = DEFAULT_POLICY, n_action_steps: int = 10, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = device
        # The released checkpoint ships n_action_steps=50 (== chunk size), which costs ~20pt on LIBERO
        # (lerobot issue #4614). Replan every 10 steps instead.
        cfg.n_action_steps = n_action_steps
        env_cfg = LiberoEnvConfig(task="libero_goal", observation_height=OBS_SIZE, observation_width=OBS_SIZE)
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    @torch.inference_mode()
    def act(self, sess: ChainSession, instruction: str) -> np.ndarray:
        obs = preprocess_observation(sess.batched_obs())
        obs["task"] = [instruction]
        obs = self.env_pre(obs)
        obs = self.pre(obs)
        action = self.post(self.policy.select_action(obs))
        action = self.env_post({"action": action})["action"]
        return action.to("cpu").numpy()[0]

    def run_skill(
        self,
        sess: ChainSession,
        instruction: str,
        max_steps: int = 300,
        stop_on_success: bool = True,
        record_every: int = 1,
        should_stop: Callable[[int], bool] | None = None,
        grace_steps: int = 0,
    ) -> SkillResult:
        """Execute one skill from the *current* scene state (no reset).

        stop_on_success: stop as soon as the simulator predicate fires (oracle).
        should_stop:     external stop test called after each step (e.g. periodic VLM check).
        grace_steps:     keep running the policy this long after stopping is decided, so it can
                         finish the motion (e.g. release the object) instead of freezing mid-place.
        """
        self.policy.reset()  # clear the action queue from the previous skill
        frames: list[np.ndarray] = []
        success = sess.is_success()
        step = 0
        stopped = stop_on_success and success
        while step < max_steps and not stopped:
            success = sess.step(self.act(sess, instruction))
            step += 1
            if step % record_every == 0:
                frames.append(sess.frame())
            stopped = (stop_on_success and success) or bool(should_stop and should_stop(step))
        for _ in range(grace_steps if stopped else 0):
            success = sess.step(self.act(sess, instruction))
            step += 1
            if step % record_every == 0:
                frames.append(sess.frame())
        return SkillResult(instruction, step, success, stopped, frames)
