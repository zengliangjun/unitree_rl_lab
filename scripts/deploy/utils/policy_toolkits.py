
import torch
from configs import configs

def load(cfg: configs.MujocoConfig):
    return torch.jit.load(cfg.policy_path)
