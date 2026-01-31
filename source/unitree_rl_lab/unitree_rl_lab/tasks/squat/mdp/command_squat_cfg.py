from __future__ import annotations
from dataclasses import MISSING
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, SceneEntityCfg

from . import command_squat

@configclass
class SquatCommandCfg(CommandTermCfg):

    class_type: type = command_squat.SquatCommand
    asset_cfg: SceneEntityCfg = MISSING

    rel_reset_init_envs: float = 0.5
    rel_compute_init_envs: float = 0.2
    rel_compute_max_envs: float = 0.2

    @configclass
    class Ranges:
        # knee_pos: tuple[float, float] = MISSING
        squat_phase: tuple[float, float] = MISSING
        full_times: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    max_limit_ranges: Ranges = MISSING
    min_limit_ranges: Ranges = MISSING

