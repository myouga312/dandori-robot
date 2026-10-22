"""LIBERO session that keeps one physical scene alive across several skills.

LIBERO-Goal's 10 tasks share one kitchen scene and differ only in the BDDL goal.
Instead of resetting between skills, we keep the simulator running and swap the
goal predicates, so the ground-truth `check_success()` follows the current skill.
(Rebuilding an env per task would re-sample fixture placement on reset.)
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from robosuite.controllers import load_controller_config
from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

OBS_SIZE = 360  # lerobot's LIBERO eval default; keep train/eval rendering identical


def _axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def _batch(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _batch(v) for k, v in tree.items()}
    if isinstance(tree, np.ndarray):
        return tree[None]
    return tree


class ChainSession:
    def __init__(self, suite_name: str = "libero_goal", init_task_id: int = 0, episode_index: int = 0):
        self.suite_name = suite_name
        self.suite = _get_suite(suite_name)
        self.env = LiberoEnv(
            task_suite=self.suite,
            task_id=init_task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            observation_width=OBS_SIZE,
            observation_height=OBS_SIZE,
            episode_index=episode_index,
        )
        self.task_id = init_task_id
        self.obs, _ = self.env.reset()
        # Chained skills run far past the suite's per-task horizon; don't let robosuite end the episode.
        self.env._env.env.ignore_done = True
        self.home_pos = self.eef_pos().copy()
        self.home_mat = self.eef_mat().copy()
        self.home_joints = self.joints().copy()

    def eef_pos(self) -> np.ndarray:
        return np.asarray(self.obs["robot_state"]["eef"]["pos"], dtype=np.float64)

    def joints(self) -> np.ndarray:
        return np.asarray(self.obs["robot_state"]["joints"]["pos"], dtype=np.float64)

    def eef_mat(self) -> np.ndarray:
        # controller's ee_ori_mat: the same frame OSC applies delta rotations in
        return np.asarray(self.obs["robot_state"]["eef"]["mat"], dtype=np.float64)

    def go_home(self, max_steps: int = 300, tol: float = 0.02, release_steps: int = 15, lift: float = 0.05,
                frames: list | None = None) -> int:
        """Scripted return to the episode's initial arm configuration, gripper open.

        Skills were trained from that start pose; ending one skill elsewhere (wrist twisted,
        elbow flipped) puts the next skill out of distribution. Task-space (OSC) homing can get
        stuck in a bad elbow configuration, so we temporarily swap the arm controller to
        JOINT_POSITION and servo the 7 joints back, then restore the original OSC controller.
        """
        domain = self.env._env.env
        robot = domain.robots[0]
        steps = 0

        def act(a: np.ndarray) -> None:
            nonlocal steps
            self.step(a)
            steps += 1
            if frames is not None:
                frames.append(self.frame())

        # 1) release: a skill may be stopped while the object is still held
        for _ in range(release_steps):
            act(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        # 2) back straight up in task space so the arm doesn't sweep objects it just placed
        z_goal = self.eef_pos()[2] + lift
        for _ in range(30):
            dz = z_goal - self.eef_pos()[2]
            if dz < 0.005:
                break
            act(np.array([0, 0, np.clip(dz / 0.05 * 2.0, 0, 1), 0, 0, 0, -1], dtype=np.float32))
        # 3) joint-space servo to the start configuration
        osc_cfg = robot.controller_config
        robot.controller_config = load_controller_config(default_controller="JOINT_POSITION")
        robot.controller_config["kp"] = 300
        robot._load_controller()
        domain._action_dim = robot.action_dim  # env caches the action size at reset
        try:
            while steps < max_steps:
                err = self.home_joints - self.joints()
                if np.abs(err).max() < tol:
                    break
                a = np.zeros(8, dtype=np.float32)
                a[:7] = np.clip(err / 0.05, -1.0, 1.0)
                a[7] = -1.0
                act(a)
        finally:
            robot.controller_config = osc_cfg
            robot._load_controller()
            robot.controller.use_delta = True
            domain._action_dim = robot.action_dim
        return steps

    # ---- task / goal switching -------------------------------------------------
    def instructions(self) -> list[str]:
        return [self.suite.get_task(i).language for i in range(self.suite.n_tasks)]

    def task_id_of(self, instruction: str) -> int:
        return self.instructions().index(instruction)

    def _bddl_path(self, task_id: int) -> str:
        t = self.suite.get_task(task_id)
        return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)

    def set_goal(self, task_id: int) -> None:
        """Swap the success predicates to `task_id` without touching physics."""
        inner = self.env._env.env  # robosuite domain holding parsed_problem
        parsed = robosuite_parse_problem(self._bddl_path(task_id))
        cur_objs = set(inner.parsed_problem["objects"]) | set(inner.parsed_problem["fixtures"])
        new_objs = set(parsed["objects"]) | set(parsed["fixtures"])
        if cur_objs != new_objs:
            raise ValueError(f"task {task_id} uses a different scene: {cur_objs ^ new_objs}")
        inner.parsed_problem["goal_state"] = parsed["goal_state"]
        inner.parsed_problem["language_instruction"] = parsed["language_instruction"]
        if "obj_of_interest" in parsed:
            inner.parsed_problem["obj_of_interest"] = parsed["obj_of_interest"]
        self.task_id = task_id

    # ---- stepping --------------------------------------------------------------
    def is_success(self) -> bool:
        return bool(self.env._env.check_success())

    def step(self, action: np.ndarray) -> bool:
        raw, _, _, _ = self.env._env.step(action)
        self.obs = self.env._format_raw_obs(raw)
        return self.is_success()

    def settle(self, n: int = 10) -> None:
        for _ in range(n):
            self.step(np.asarray(get_libero_dummy_action(), dtype=np.float32))

    def batched_obs(self) -> dict:
        return _batch(self.obs)

    # ---- rendering (upright, as a human sees the scene) -------------------------
    def frame(self) -> np.ndarray:
        return np.ascontiguousarray(self.obs["pixels"]["image"][::-1, ::-1])

    def wrist(self) -> np.ndarray:
        return np.ascontiguousarray(self.obs["pixels"]["image2"][::-1, ::-1])

    def sim_state(self) -> np.ndarray:
        return self.env._env.sim.get_state().flatten()

    def close(self) -> None:
        self.env.close()
