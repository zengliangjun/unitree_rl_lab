import math
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .actions import MotionActions


@configclass
class MotionsActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MotionActions

    motions_dir: str = MISSING
    joint_names: list[str] = MISSING
    scale: float = 1

    preserve_order: bool = True
    use_default_offset: bool = True
