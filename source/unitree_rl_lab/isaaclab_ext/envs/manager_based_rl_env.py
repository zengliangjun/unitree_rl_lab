from isaaclab.envs import ManagerBasedRLEnv

from .manager_based_rl_env_cfg import ModuleRLEnvCfg
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager
from isaaclab_ext.managers import reward_manager

class ModuleRLEnv(ManagerBasedRLEnv):

    cfg: ModuleRLEnvCfg

    def __init__(self, cfg: ModuleRLEnvCfg, render_mode: str | None = None, **kwargs):
        super(ModuleRLEnv, self).__init__(cfg, render_mode, **kwargs)


    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super(ManagerBasedRLEnv, self).load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = reward_manager.RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
