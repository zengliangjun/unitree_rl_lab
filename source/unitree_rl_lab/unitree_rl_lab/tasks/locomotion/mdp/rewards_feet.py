from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from enum import Enum
class FeetStatus(Enum):
    NONE = 0
    Contact_Status = 1
    Air_Status = 2

class TimesSymmetry(ManagerTermBase):
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self.sensor_cfg = sensor_cfg
        self.command_name = cfg.params["command_name"]

        self._init_buffers()
        # 初始化标志位

    def _init_buffers(self):
        # 初始化足接触统计的均值与方差缓冲区 (一维数据)
        posecount = len(self.sensor_cfg.body_ids)

        self.feet_status = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.int8)

        self.air_steps_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.int32)

        self.air_variance_buf = torch.zeros((self.num_envs, posecount),
                              device=self.device, dtype=torch.float)
        self.air_mean_buf = torch.zeros_like(self.air_variance_buf)

        self.contact_steps_buf = torch.zeros_like(self.air_steps_buf)

        self.contact_variance_buf = torch.zeros_like(self.air_variance_buf)
        self.contact_mean_buf = torch.zeros_like(self.air_variance_buf)



    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 重置足接触统计的缓冲区，并导出导数数据
        if env_ids is None or len(env_ids) == 0:
            return

        # 清空所有足接触统计缓冲区
        buffers = [
            self.feet_status,
            self.air_steps_buf,
            self.air_variance_buf,
            self.air_mean_buf,

            self.contact_steps_buf,
            self.contact_variance_buf,
            self.contact_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

    def _update_flag(self):
        command: UniformVelocityCommand = self._env.command_manager.get_term(self.command_name)
        self.stand_flag = torch.logical_or(command.is_standing_env ,
                                           self._env.episode_length_buf <= 1)

    def _calcute_time(self, diff: torch.Tensor, mask: torch.Tensor, air_or_contact: bool) -> None:
        # 利用增量更新方法计算当前episode的均值和方差
        if air_or_contact:
            steps_buf = self.air_steps_buf
            variance_buf = self.air_variance_buf
            mean_buf = self.air_mean_buf
        else:
            steps_buf = self.contact_steps_buf
            variance_buf = self.contact_variance_buf
            mean_buf = self.contact_mean_buf

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = diff - mean_buf
        work_mean_buf = mean_buf + delta0 / steps_buf

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = diff - work_mean_buf
        work_variance_buf = (
            variance_buf * (steps_buf - 2)
            + delta0 * delta1
        ) / (steps_buf - 1)

        mean_buf[mask] = work_mean_buf[mask]
        variance_buf[mask] = work_variance_buf[mask]

        # 当episode刚开始时重置方差，防止数值异常
        new_mask = steps_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        variance_buf[new_mask] = 0

    def _calcute_contact(self):

        air_time = self.contact_sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        contact_time = self.contact_sensor.data.last_contact_time[:, self.sensor_cfg.body_ids]
        contact = self.contact_sensor.data.current_contact_time[:, self.sensor_cfg.body_ids]
        in_contact = contact > 0.0

        init_mask = self.feet_status == FeetStatus.NONE.value
        air_mask = self.feet_status == FeetStatus.Air_Status.value
        contact_mask = self.feet_status == FeetStatus.Contact_Status.value


        init_flags = torch.logical_and(in_contact, init_mask)
        update2contact = torch.logical_and(in_contact, air_mask)
        update2air = torch.logical_and(torch.logical_not(in_contact), contact_mask)

        self.feet_status[init_flags] = FeetStatus.Contact_Status.value
        self.feet_status[update2air] = FeetStatus.Air_Status.value
        self.feet_status[update2contact] = FeetStatus.Contact_Status.value

        self.air_steps_buf[update2contact] += 1
        self.contact_steps_buf[update2air] += 1

        self._calcute_time(air_time, update2contact, True)
        self._calcute_time(contact_time, update2air, False)

    def _calculate(self, error_std: float = 0.06
        ) -> torch.Tensor:

        air_mean = torch.abs(self.air_mean_buf[:, 0] - self.air_mean_buf[:, 1])
        air_var = torch.norm(self.air_variance_buf, dim = -1)

        rew = torch.exp(- air_mean / error_std) + torch.exp(- air_var / error_std)
        return rew

    def __call__(self,
            env: ManagerBasedRLEnv,
            command_name: str = "base_velocity",
            sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

            error_std: float = 0.06
        ) -> torch.Tensor:
        self._update_flag()
        self._calcute_contact()

        return self._calculate(error_std) # + self._calculate_varx(std_ranges, error_std)

