from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from unitree_rl_lab.tasks.locomotionv2.mdp import commands
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def action_rate_l2_ext(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]), dim=1)

def stand_deviation_l1(env: ManagerBasedRLEnv,
                       command_name: str = "base_velocity",
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    is_walking = torch.logical_not(cmd.is_standing_env)
    pos_error[is_walking, :] = 0.0
    return torch.sum(torch.abs(pos_error), dim=1)

def track_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name="stomp_command",
) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.005

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    is_stance = cmd.feet_global_phases < cmd.cfg.threshold

    reward = ~(is_stance ^ is_contact)
    return torch.sum(reward, dim=-1)

def penalize_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name="stomp_command",
) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.005

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    is_swing = cmd.feet_global_phases > cmd.cfg.threshold

    penalize = (is_swing ^ is_contact)
    return torch.sum(penalize, dim=-1)


def penalty_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    gw = torch.repeat_interleave(asset.data.GRAVITY_VEC_W.unsqueeze(1), repeats=len(asset_cfg.body_ids), dim=1)
    body_gravity_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], gw)
    return torch.sum(torch.sum(torch.square(body_gravity_b[:, :, :2]), dim=-1), dim=-1)


def penalty_contact_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    no_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] < 0.005

    asset: Articulation = env.scene[asset_cfg.name]
    gw = torch.repeat_interleave(asset.data.GRAVITY_VEC_W.unsqueeze(1), repeats=len(asset_cfg.body_ids), dim=1)
    body_gravity_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], gw)

    error = torch.norm(body_gravity_b[:, :, :2], dim=-1) / std
    error[no_contact] = 0

    return torch.sum(error, dim=-1)


def feet_slide(env,
    command_name: str = "stomp_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)
    stand_phases = cmd.feet_global_phases < cmd.cfg.threshold     # N * 2

    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    reward = torch.sum(body_vel.norm(dim=-1) * stand_phases.float(), dim= -1)
    return reward


def feet_slide_ang(env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]

    body_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def reward_feet_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.2,
    std: float = 0.04
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]

    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], body_pos_w.shape[1], dim=1)

    body_pos = math_utils.quat_apply_inverse(quat_w, body_pos_w)

    #error = torch.square((torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width) / std)
    error = torch.abs((torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width) / std)

    return - torch.norm(error, dim=-1) + torch.norm(torch.exp(- error), dim=-1)


def reward_feet_widthv2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.2,
    std: float = 0.04
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]

    quat_w = torch.repeat_interleave(asset.data.root_link_quat_w[:, None, :], body_pos_w.shape[1], dim=1)

    body_pos = math_utils.quat_apply_inverse(quat_w, body_pos_w)

    #error = torch.square((torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width) / std)
    error = (torch.abs(body_pos[:, 0::2, 1] - body_pos[:, 1::2, 1]) - target_width)
    iner_flag = error < 0
    out_flag = error > 0

    error[iner_flag] = torch.square(error[iner_flag] / (std / 2))
    error[out_flag] = torch.square(error[out_flag] / std)

    return - torch.norm(error, dim=-1) + torch.norm(torch.exp(- error), dim=-1)


def penalize_feet_forces(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg,
   threshold: float = 500,
   max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Compute the norm of forces for specified body parts.
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces)
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def penalize_feet_forces_v2(env: ManagerBasedRLEnv,
   sensor_cfg: SceneEntityCfg,
   threshold: float = 500,
   contact_time_threshold: float = 0.07,
   max_over_penalize_forces: float = 400) -> torch.Tensor:
    """
    Penalizes excessive contact forces on the feet to discourage high impact.

    Args:
        env (ManagerBasedRLEnv): The simulation environment instance.
        sensor_cfg (SceneEntityCfg): Sensor configuration including sensor name and body IDs.
        threshold (float): Force threshold above which penalties are applied.
        max_over_penalize_forces (float): Maximum force value to clip the penalty.

    Returns:
        torch.Tensor: The calculated penalty as the sum of clamped excessive forces from specified body parts.
    """
    # Retrieve the contact sensor instance.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_flags
    contact_flags = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] < contact_time_threshold

    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    # Calculate penalty by clamping forces exceeding the threshold.
    _reward = torch.clamp(forces - threshold, min=0, max=max_over_penalize_forces) * contact_flags.float()
    # Sum penalties across all relevant body parts.
    _reward = torch.sum(_reward, dim=-1)
    return _reward

def reward_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)

    feet_exp = torch.exp(- diff)
    return torch.mean(feet_exp, dim=-1)


def penalize_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = cmd.feet_swing_phases

    swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    swing_target = torch.clamp_min(swing_target0, min=0.0)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)
    return torch.mean(diff, dim=-1)


def reward_foot_clearance_v2(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = torch.logical_and(cmd.feet_swing_phases > 0.008, cmd.feet_swing_phases < 0.992)

    #swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    #swing_target = torch.clamp_min(swing_target0, min=0.0)
    swing_target = swing_phase.float() * target_height

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)

    feet_exp = torch.exp(- diff)
    return torch.mean(feet_exp, dim=-1)


def penalize_foot_clearance_v2(
    env: ManagerBasedRLEnv,
    command_name: str,

    std: float,
    max_height: float,
    target_height: float,
    speed: float,

    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    swing_phase = torch.logical_and(cmd.feet_swing_phases > 0.008, cmd.feet_swing_phases < 0.992)

    # swing_target0 = torch.sin(swing_phase * torch.pi) * target_height
    # swing_target = torch.clamp_min(swing_target0, min=0.0)
    swing_target = swing_phase.float() * target_height

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  ##
    # feet_error = torch.clamp_max(feet_z - target_height, max=0) # allow feet to be higher than target, but penalize if they are lower
    feet_error = feet_z - swing_target

    clamp_mask = swing_target > 0.008
    feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0) # only penalize when target height is above 0 (i.e. during swing phase)

    diff = torch.square(feet_error / std)
    return torch.mean(diff, dim=-1)


def flat_orientation_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.projected_gravity_b[:, 0])


def flat_orientation_y(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.projected_gravity_b[:, 1])

# ============================================================================
# 贝塞尔曲线奖励函数
# ============================================================================
from . import bezier

def reward_foot_clearance_bezier(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    max_height: float,
    target_height: float,
    speed: float,
    asset_cfg: SceneEntityCfg,
    control_points: list[tuple[float, float]] | None = [(0.0, 0.0), (0.25, 2), (0.8, 0.4), (1.0, 0.0)],
    use_clamp: bool = True,
    swing_phase_threshold: float = 0.003
) -> torch.Tensor:
    """
    使用贝塞尔曲线奖励摆动脚离地高度（参数化版本）

    通过贝塞尔曲线生成平滑的脚部轨迹目标，奖励脚部实际高度接近目标高度。

    Args:
        env: 环境实例
        command_name: 命令名称
        std: 标准差，用于计算奖励
        max_height: 最大高度（未使用，为了兼容性保留）
        target_height: 目标高度
        speed: 速度参数（未使用，为了兼容性保留）
        asset_cfg: 资产配置
        control_points: 贝塞尔曲线控制点，支持多种格式：
            - None: 使用默认控制点 [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
            - 列表: [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
            - 字符串: "0.0,0.0;0.2,0.9;0.8,0.2;1.0,0.0"
        use_clamp: 是否使用clamp限制误差（只惩罚高度不足）
        swing_phase_threshold: 摆动相位阈值，低于此值认为脚在地面

    Returns:
        torch.Tensor: 奖励值
    """
    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    # 使用摆动相位
    swing_phase = cmd.feet_swing_phases

    # 计算贝塞尔曲线目标高度
    swing_target = bezier._bezier_target_height(swing_phase, target_height, control_points)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 计算高度误差
    feet_error = feet_z - swing_target

    # 如果使用clamp，只惩罚高度不足（脚低于目标）
    if use_clamp:
        clamp_mask = swing_target > swing_phase_threshold
        feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0)

    # 计算指数奖励
    diff = torch.square(feet_error / std)
    feet_exp = torch.exp(-diff)

    return torch.mean(feet_exp, dim=-1)


def penalize_foot_clearance_bezier(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    max_height: float,
    target_height: float,
    speed: float,
    asset_cfg: SceneEntityCfg,
    control_points: list[tuple[float, float]] | None = [(0.0, 0.0), (0.25, 2), (0.8, 0.4), (1.0, 0.0)],
    use_clamp: bool = True,
    swing_phase_threshold: float = 0.003
) -> torch.Tensor:
    """
    使用贝塞尔曲线惩罚摆动脚离地高度误差（参数化版本）

    通过贝塞尔曲线生成平滑的脚部轨迹目标，惩罚脚部实际高度与目标高度的偏差。

    Args:
        env: 环境实例
        command_name: 命令名称
        std: 标准差，用于计算惩罚
        max_height: 最大高度（未使用，为了兼容性保留）
        target_height: 目标高度
        speed: 速度参数（未使用，为了兼容性保留）
        asset_cfg: 资产配置
        control_points: 贝塞尔曲线控制点，支持多种格式：
            - None: 使用默认控制点 [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
            - 列表: [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]
            - 字符串: "0.0,0.0;0.2,0.9;0.8,0.2;1.0,0.0"
        use_clamp: 是否使用clamp限制误差（只惩罚高度不足）
        swing_phase_threshold: 摆动相位阈值，低于此值认为脚在地面

    Returns:
        torch.Tensor: 惩罚值
    """
    cmd: commands.CommandWithPhase = env.command_manager.get_term(command_name)

    # 使用摆动相位
    swing_phase = cmd.feet_swing_phases

    # 计算贝塞尔曲线目标高度
    swing_target = bezier._bezier_target_height(swing_phase, target_height, control_points)

    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 计算高度误差
    feet_error = feet_z - swing_target

    # 如果使用clamp，只惩罚高度不足（脚低于目标）
    if use_clamp:
        clamp_mask = swing_target > swing_phase_threshold
        feet_error[clamp_mask] = torch.clamp_max(feet_error[clamp_mask], max=0)

    # 计算平方误差惩罚
    diff = torch.square(feet_error / std)

    return torch.mean(diff, dim=-1)


# ============================================================================
# 预设控制点配置
# ============================================================================

# 默认三次贝塞尔曲线（快速上升，缓慢下降）
DEFAULT_CUBIC_BEZIER_POINTS = [(0.0, 0.0), (0.2, 0.9), (0.8, 0.2), (1.0, 0.0)]

# 二次贝塞尔曲线（对称抛物线）
DEFAULT_QUADRATIC_BEZIER_POINTS = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]

# 四次贝塞尔曲线（更复杂的轨迹）
DEFAULT_QUARTIC_BEZIER_POINTS = [(0.0, 0.0), (0.2, 0.7), (0.4, 1.0), (0.6, 0.8), (1.0, 0.0)]

# 线性插值（与v2版本类似，但平滑）
DEFAULT_LINEAR_POINTS = [(0.0, 0.0), (1.0, 0.0)]

# 正弦曲线近似（与原始版本类似）
DEFAULT_SINUSOIDAL_POINTS = [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 0.5), (1.0, 0.0)]

