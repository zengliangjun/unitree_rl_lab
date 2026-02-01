
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from loguru import logger as ulogger

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .command_squat_cfg import SquatCommandCfg


class SquatCommand(CommandTerm):
    """
    下蹲动作命令生成器。

    这个类负责生成机器人下蹲动作的命令，使用相位（phase）来表示关节位置。
    关节位置通过正弦函数映射到相位空间：pos = cpos - sin(phase) * rad

    相位图：
      + pi / 2
      |
      |
      +-------- 0
      |
      |
      - pi / 2

    其中：
    - cpos: 关节位置的中心点（关节运动范围的中点）
    - rad: 关节运动范围的半径（半幅值）
    - phase: 相位角度，控制关节位置

    主要功能：
    1. 生成目标相位命令
    2. 计算当前相位位置
    3. 控制相位变化速度
    4. 管理环境状态（初始化、最大位置、完成标志）
    5. 跟踪性能指标
    """

    cfg: SquatCommandCfg

    def __init__(self, cfg: SquatCommandCfg, env: ManagerBasedRLEnv):
        """
        初始化下蹲命令生成器。

        Args:
            cfg: 配置对象，包含命令生成器的参数
            env: RL环境对象
        """
        super().__init__(cfg, env)

        # 获取机器人资产（关节）
        self.asset: Articulation = env.scene[cfg.asset_cfg.name]
        cfg.asset_cfg.resolve(env.scene)

        # 初始化状态缓冲区
        self.squat_phase = torch.zeros(self.num_envs, 1, device=self.device)  # 当前相位
        self.squat_command_phase = torch.zeros(self.num_envs, 1, device=self.device)  # 目标相位
        self.squat_phase_vel = torch.zeros(self.num_envs, 1, device=self.device)  # 相位速度
        self.is_init_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否为初始化环境
        self.is_max_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否为最大位置环境
        self.is_finished_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否完成标志
        self.episode_length_buffer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 回合长度缓冲区
        self.average_episode_length = 0  # 平均回合长度

        # 计算关节位置参数
        # pos_limits: 形状为 (num_envs, num_joints, 2)，包含每个关节的最小和最大位置限制
        pos_limits = self.asset.data.joint_pos_limits[:, self.cfg.asset_cfg.joint_ids].clone()

        pos_min = pos_limits[:, :, 0]  #
        pos_max = pos_limits[:, :, 1]

        # cpos: 关节位置中心点，计算为关节位置限制的平均值
        # 形状: (num_envs, 1)
        self.cpos = torch.mean((pos_min + pos_max) / 2, dim=-1, keepdim=True)

        # rad: 关节运动半径，计算为关节位置范围的一半
        # 形状: (num_envs, 1)
        self.rad = torch.mean((pos_max - pos_min) / 2, dim=-1, keepdim=True)

        # 计算初始相位：基于默认关节位置
        # 使用正弦逆函数将位置映射到相位：phase = asin((cpos - pos) / rad)
        pos = torch.mean(self.asset.data.default_joint_pos[:, self.cfg.asset_cfg.joint_ids], dim=-1, keepdim=True)

        sin = (self.cpos - pos) / self.rad
        sin = torch.clamp(sin, -1.0, 1.0)
        self.init_phase = torch.asin(sin)

        # 初始化性能指标
        self.metrics["error_knee_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """
        返回命令生成器的字符串表示。

        Returns:
            包含命令生成器信息的字符串
        """
        msg = "SquatCommand:\n"
        msg += f"\tTarget dimension: {tuple(self.squat_phase.shape[1:])}\n"
        msg += f"\tCommand dimension: {tuple(self.squat_command_phase.shape[1:])}\n"
        msg += f"\tvel dimension: {tuple(self.squat_phase_vel.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        获取当前命令。

        命令包含以下部分：
        1. 归一化的相位命令和当前相位（除以π）
        2. 相位的余弦值
        3. 相位的正弦值
        4. 相位速度

        Returns:
            命令张量，形状为 (num_envs, 5)
        """
        com = torch.cat((self.squat_command_phase, self.squat_phase), dim=-1)
        com_cos = torch.cos(com)
        com_sin = torch.sin(com)
        result = torch.cat((com / torch.pi, com_cos, com_sin, self.squat_phase_vel), dim=-1)

        isnan = torch.isnan(result)
        if torch.sum(isnan.float()) > 0:
            print(result[isnan])
            raise ValueError(f"command {torch.sum(isnan.float())}. error")
        return result

    @property
    def command_pos(self) -> torch.Tensor:
        """
        根据当前相位计算关节位置。

        使用公式：pos = cpos - sin(phase) * rad

        Returns:
            关节位置张量，形状为 (num_envs, num_joints)
        """
        pos = self.cpos - torch.sin(self.squat_phase) * self.rad

        isnan = torch.isnan(pos)
        if torch.sum(isnan.float()) > 0:
            print(pos[isnan])
            raise ValueError(f"command_pos {torch.sum(isnan.float())}. error")

        return pos

    def compute(self, dt: float):
        """
        计算命令。

        主要步骤：
        1. 更新性能指标
        2. 减少重新采样时间
        3. 如果需要，重新采样命令
        4. 更新命令状态

        Args:
            dt: 自上次调用compute以来经过的时间步长
        """
        # 基于当前状态更新指标
        self._update_metrics()
        # 减少重新采样前剩余时间
        self.time_left -= dt
        # 如果需要，重新采样命令
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            # 修复类型错误：将张量转换为列表
            self._resample_compute(resample_env_ids.tolist())
        # 更新命令
        self._update_command()


    def _resample_compute(self, env_ids: Sequence[int]):
        """
        重新采样命令（用于compute方法中的重新采样）。

        这个函数为指定的环境索引重新采样命令和应用时间。

        Args:
            env_ids: 要重新采样的环境ID列表
        """
        if len(env_ids) != 0:
            # 重新采样重新采样前的剩余时间
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)

            # 清理标志
            self.is_init_env[env_ids] = 0
            self.is_max_env[env_ids] = 0
            self.is_finished_flags[env_ids] = 0
            # 重新采样命令
            self._resample_command_compute(env_ids)
            # 增加命令计数器
            self.command_counter[env_ids] += 1

    def _resample_command_compute(self, env_ids: Sequence[int]):
        """
        重新采样命令参数（用于compute方法）。

        为指定的环境重新采样：
        1. 相位速度（基于完整时间）
        2. 目标相位命令
        3. 环境状态（初始化、最大位置）
        4. 当前相位

        Args:
            env_ids: 要重新采样的环境ID列表
        """
        # 创建随机数生成器
        r = torch.empty(len(env_ids), device=self.device)

        # 采样完整时间并计算相位速度
        # 相位速度 = π / 完整时间，表示完成半个周期所需的速度
        full_times = r.uniform_(*self.cfg.ranges.full_times)
        self.squat_phase_vel[env_ids, 0] = torch.pi / full_times

        # 更新目标相位命令
        self.squat_command_phase[env_ids, 0] = r.uniform_(*self.cfg.ranges.squat_phase)

        # 确定环境状态
        uniform_prop = r.uniform_(0.0, 1.0)
        self.is_init_env[env_ids] = uniform_prop <= self.cfg.rel_compute_init_envs
        self.is_max_env[env_ids] = torch.logical_and(
            uniform_prop > self.cfg.rel_compute_init_envs,
            uniform_prop <= self.cfg.rel_compute_init_envs + self.cfg.rel_compute_max_envs
        )

        # 根据环境状态调整目标相位
        # 初始化环境：使用初始相位
        self.squat_command_phase[self.is_init_env] = self.init_phase[self.is_init_env]
        # 最大位置环境：使用最小位置（最大下蹲深度）
        self.squat_command_phase[self.is_max_env] = self.cfg.ranges.squat_phase[0]

        # 获取当前关节位置并计算当前相位
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]
        # 使用正弦逆函数计算相位：phase = asin((cpos - pos) / rad)
        phase_input = (self.cpos[env_ids, :] - pos[env_ids, :]) / self.rad[env_ids, :]
        # 添加数值稳定性保护
        phase_input = torch.clamp(phase_input, -1.0, 1.0)
        phase = torch.asin(phase_input)
        self.squat_phase[env_ids, :] = phase.mean(dim=-1, keepdim=True)

        # 调整相位速度的方向：朝向目标相位
        # 如果目标相位 > 当前相位，速度为正；否则为负
        self.squat_phase_vel[env_ids, :] *= torch.sign(
            self.squat_command_phase[env_ids] - self.squat_phase[env_ids]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        重新采样命令（用于环境重置）。

        这个函数在环境重置时被调用，重新采样命令参数并更新平均回合长度。

        Args:
            env_ids: 要重新采样的环境ID列表
        """
        # 更新平均回合长度（指数移动平均）
        episode_length_mean = torch.mean(self.episode_length_buffer[env_ids].float()).cpu().item()
        self.average_episode_length = self.average_episode_length * 0.9 + episode_length_mean * 0.1

        # 重置回合长度缓冲区
        self.episode_length_buffer[env_ids] = 0

        # 清理标志
        self.is_init_env[env_ids] = 0
        self.is_max_env[env_ids] = 0
        self.is_finished_flags[env_ids] = 0

        # 创建随机数生成器
        r = torch.empty(len(env_ids), device=self.device)

        # 采样完整时间并计算相位速度
        full_times = r.uniform_(*self.cfg.ranges.full_times)
        self.squat_phase_vel[env_ids, 0] = torch.pi / full_times

        # 更新目标相位命令
        self.squat_command_phase[env_ids, 0] = r.uniform_(*self.cfg.ranges.squat_phase)

        # 确定初始化环境
        self.is_init_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_reset_init_envs
        self.squat_command_phase[self.is_init_env] = self.init_phase[self.is_init_env]

        # 获取当前关节位置并计算当前相位
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]
        # 使用正弦逆函数计算相位，添加数值稳定性保护
        phase_input = (self.cpos[env_ids, :] - pos[env_ids, :]) / self.rad[env_ids, :]
        phase_input = torch.clamp(phase_input, -1.0, 1.0)
        phase = torch.asin(phase_input)
        self.squat_phase[env_ids, :] = phase.mean(dim=-1, keepdim=True)

        # 调整相位速度的方向：朝向目标相位
        self.squat_phase_vel[env_ids, :] *= torch.sign(
            self.squat_command_phase[env_ids] - self.squat_phase[env_ids]
        )

    def _update_command(self):
        """
        更新命令状态。

        主要步骤：
        1. 增加回合长度计数器
        2. 更新当前相位（根据相位速度和时间步长）
        3. 检查是否到达目标相位，更新完成标志
        4. 限制相位在目标范围内
        """
        # 增加回合长度计数器
        self.episode_length_buffer += 1

        # 更新当前相位：phase = phase + velocity * dt
        self.squat_phase += self.squat_phase_vel * self._env.step_dt

        # 处理正向速度（phase_vel >= 0）的环境
        # 这些环境正在向更大的相位值移动
        vel_flag_positive = self.squat_phase_vel >= 0
        if torch.sum(vel_flag_positive) > 0:
            flag = vel_flag_positive[:, 0]
            # 检查是否到达或超过目标相位
            finished_flags = self.squat_phase[flag] >= self.squat_command_phase[flag]
            self.is_finished_flags[flag] = finished_flags[:, 0]
            # 限制相位不超过目标相位
            self.squat_phase[vel_flag_positive] = torch.clamp_max(
                self.squat_phase[vel_flag_positive],
                self.squat_command_phase[vel_flag_positive]
            )

        # 处理负向速度（phase_vel < 0）的环境
        # 这些环境正在向更小的相位值移动
        vel_flag_negative = self.squat_phase_vel < 0
        if torch.sum(vel_flag_negative) > 0:
            flag = vel_flag_negative[:, 0]
            # 检查是否到达或低于目标相位
            finished_flags = self.squat_phase[flag] <= self.squat_command_phase[flag]
            self.is_finished_flags[flag] = finished_flags[:, 0]
            # 限制相位不低于目标相位
            self.squat_phase[vel_flag_negative] = torch.clamp_min(
                self.squat_phase[vel_flag_negative],
                self.squat_command_phase[vel_flag_negative]
            )

    def _update_metrics(self):
        """
        更新性能指标。

        计算当前膝关节位置与命令位置之间的误差，并累加到指标中。
        """
        # 获取当前关节位置
        pos = self.asset.data.joint_pos[:, self.cfg.asset_cfg.joint_ids]

        # 计算位置误差并累加
        # 使用欧几里得距离（L2范数）计算误差
        position_error = torch.norm(self.command_pos - pos, dim=-1)
        self.metrics["error_knee_pos"] += position_error
