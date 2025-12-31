from __future__ import annotations
from dataclasses import MISSING

from .commands import UniformLevelVelocityCommandCfg
from .commands_ext import CommandExt

from isaaclab.utils import configclass

@configclass
class CommandExtCfg(UniformLevelVelocityCommandCfg):

    class_type: type = CommandExt

    # rel_standing_envs
    # rel_standing_envs: float = 0.1
    rel_walking_envs: float = 0.23
    rel_spining_envs: float = 0.23
    rel_walking_spining_envs: float = 0.44

    def __post_init__(self) -> None:
        super().__post_init__()
        if not abs(self.rel_standing_envs + self.rel_walking_envs + self.rel_spining_envs + self.rel_walking_spining_envs - 1.0) < 1e-6:
            raise ValueError("The sum of rel_standing_envs, rel_walking_envs, rel_spining_envs, and rel_walking_spining_envs must be 1.0")

        self.heading_command = False

