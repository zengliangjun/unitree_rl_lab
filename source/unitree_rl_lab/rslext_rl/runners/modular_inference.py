from torch import nn
import torch
from rsl_rl.modules import ActorCritic
import copy
import os
import os.path as osp

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter, _TorchPolicyExporter


class OnnxExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, leg_policy: ActorCritic, arm_policy: ActorCritic, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.leg = _OnnxPolicyExporter(leg_policy, verbose=verbose)
        self.arm = _OnnxPolicyExporter(arm_policy, verbose=verbose)

        if self.leg.is_recurrent and self.arm.is_recurrent:
            if self.leg.rnn_type == "lstm" and self.arm.rnn_type == "lstm":
                self.forward = self.forward_lstm_lstm
            if self.leg.rnn_type == "lstm" and self.arm.rnn_type == "gru":
                self.forward = self.forward_lstm_gru
            if self.leg.rnn_type == "gru" and self.arm.rnn_type == "gru":
                self.forward = self.forward_gru_gru
            if self.leg.rnn_type == "gru" and self.arm.rnn_type == "lstm":
                self.forward = self.forward_gru_lstm

        elif self.leg.is_recurrent:
            if self.leg.rnn_type == "lstm":
                self.forward = self.forward_lstm_norm
            if self.leg.rnn_type == "gru":
                self.forward = self.forward_gru_norm

        elif self.arm.is_recurrent:
            if self.arm.rnn_type == "lstm":
                self.forward = self.forward_norm_lstm
            if self.arm.rnn_type == "gru":
                self.forward = self.forward_norm_gru
        else:
            # self.forward = self.forward_norm_gru
            pass

    def forward(self, obs):
        legactions = self.leg(obs)
        armactions = self.arm(obs)
        return torch.cat((legactions, armactions), dim = -1)

    def forward_lstm_lstm(self, obs, leg_h_in, leg_c_in, arm_h_in, arm_c_in):

        legactions, leg_h_out, leg_c_out = self.leg(obs, leg_h_in, leg_c_in)
        armactions, arm_h_out, arm_c_out = self.arm(obs, arm_h_in, arm_c_in)
        return torch.cat(legactions, armactions), leg_h_out, leg_c_out, arm_h_out, arm_c_out

    def forward_lstm_gru(self, obs, leg_h_in, leg_c_in, arm_h_in):
        legactions, leg_h_out, leg_c_out = self.leg(obs, leg_h_in, leg_c_in)
        armactions, arm_h_out = self.arm(obs, arm_h_in)
        return torch.cat(legactions, armactions), leg_h_out, leg_c_out, arm_h_out

    def forward_gru_gru(self, obs, leg_h_in, arm_h_in):
        legactions, leg_h_out = self.leg(obs, leg_h_in)
        armactions, arm_h_out = self.arm(obs, arm_h_in)
        return torch.cat(legactions, armactions), leg_h_out, arm_h_out

    def forward_gru_lstm(self, obs, leg_h_in, arm_h_in, arm_c_in):

        legactions, leg_h_out = self.leg(obs, leg_h_in)
        armactions, arm_h_out, arm_c_out = self.arm(obs, arm_h_in, arm_c_in)
        return torch.cat(legactions, armactions), leg_h_out, arm_h_out, arm_c_out

    def forward_lstm_norm(self, obs, leg_h_in, leg_c_in):
        legactions, leg_h_out, leg_c_out = self.leg(obs, leg_h_in, leg_c_in)
        armactions = self.arm(obs)
        return torch.cat(legactions, armactions), leg_h_out, leg_c_out

    def forward_gru_norm(self, obs, leg_h_in):
        legactions, leg_h_out = self.leg(obs, leg_h_in)
        armactions = self.arm(obs)
        return torch.cat(legactions, armactions), leg_h_out

    def forward_norm_lstm(self, obs, arm_h_in, arm_c_in):
        legactions = self.leg(obs)
        armactions, arm_h_out, arm_c_out = self.arm(obs, arm_h_in, arm_c_in)
        return torch.cat(legactions, armactions), arm_h_out, arm_c_out

    def forward_norm_gru(self, obs, arm_h_in):
        legactions = self.leg(obs)
        armactions, arm_h_out = self.arm(obs, arm_h_in)
        return torch.cat(legactions, armactions), arm_h_out

    def export(self, filename):
        self.to("cpu")
        self.eval()

        if not self.leg.is_recurrent and not self.arm.is_recurrent:
            obs = torch.zeros(1, self.leg.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                filename,
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )

        else:
            if self.leg.is_recurrent and self.arm.is_recurrent:

                obs = torch.zeros(1, self.leg.rnn.input_size)
                leg_h_in = torch.zeros(self.leg.rnn.num_layers, 1, self.leg.rnn.hidden_size)
                arm_h_in = torch.zeros(self.arm.rnn.num_layers, 1, self.arm.rnn.hidden_size)

                if self.leg.rnn_type == "lstm" and self.arm.rnn_type == "lstm":

                    leg_c_in = torch.zeros(self.leg.rnn.num_layers, 1, self.leg.rnn.hidden_size)
                    arm_c_in = torch.zeros(self.arm.rnn.num_layers, 1, self.arm.rnn.hidden_size)

                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, leg_c_in, arm_h_in, arm_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in", "leg_c_in", "arm_h_in", "arm_c_in"],
                        output_names=["actions", "leg_h_out", "leg_c_out", "arm_h_out", "arm_c_out"],
                        dynamic_axes={},
                    )

                if self.leg.rnn_type == "lstm" and self.arm.rnn_type == "gru":
                    leg_c_in = torch.zeros(self.leg.rnn.num_layers, 1, self.leg.rnn.hidden_size)

                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, leg_c_in, arm_h_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in", "leg_c_in", "arm_h_in"],
                        output_names=["actions", "leg_h_out", "leg_c_out", "arm_h_out"],
                        dynamic_axes={},
                    )

                if self.leg.rnn_type == "gru" and self.arm.rnn_type == "gru":
                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, arm_h_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in", "arm_h_in"],
                        output_names=["actions", "leg_h_out", "arm_h_out"],
                        dynamic_axes={},
                    )

                if self.leg.rnn_type == "gru" and self.arm.rnn_type == "lstm":
                    arm_c_in = torch.zeros(self.arm.rnn.num_layers, 1, self.arm.rnn.hidden_size)

                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, arm_h_in, arm_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in", "arm_h_in", "arm_c_in"],
                        output_names=["actions", "leg_h_out", "arm_h_out", "arm_c_out"],
                        dynamic_axes={},
                    )

            elif self.leg.is_recurrent:

                obs = torch.zeros(1, self.leg.rnn.input_size)
                leg_h_in = torch.zeros(self.leg.rnn.num_layers, 1, self.leg.rnn.hidden_size)

                if self.leg.rnn_type == "lstm":
                    leg_c_in = torch.zeros(self.leg.rnn.num_layers, 1, self.leg.rnn.hidden_size)

                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, leg_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in", "leg_c_in"],
                        output_names=["actions", "leg_h_out", "leg_c_out"],
                        dynamic_axes={},
                    )

                if self.leg.rnn_type == "gru":

                    torch.onnx.export(
                        self,
                        (obs, leg_h_in, leg_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "leg_h_in"],
                        output_names=["actions", "leg_h_out"],
                        dynamic_axes={},
                    )

            elif self.arm.is_recurrent:

                obs = torch.zeros(1, self.arm.rnn.input_size)
                arm_h_in = torch.zeros(self.arm.rnn.num_layers, 1, self.arm.rnn.hidden_size)

                if self.arm.rnn_type == "lstm":

                    arm_c_in = torch.zeros(self.arm.rnn.num_layers, 1, self.arm.rnn.hidden_size)

                    torch.onnx.export(
                        self,
                        (obs, arm_h_in, arm_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "arm_h_in", "arm_c_in"],
                        output_names=["actions", "arm_h_out", "arm_c_out"],
                        dynamic_axes={},
                    )

                if self.arm.rnn_type == "gru":

                    torch.onnx.export(
                        self,
                        (obs, arm_h_in, arm_c_in),
                        filename,
                        export_params=True,
                        opset_version=11,
                        verbose=self.verbose,
                        input_names=["obs", "arm_h_in"],
                        output_names=["actions", "arm_h_out"],
                        dynamic_axes={},
                    )


class ModularInference(nn.Module):

    leg_policy: ActorCritic
    arm_policy: ActorCritic

    def __init__(self, leg_policy: ActorCritic, arm_policy: ActorCritic):
        super(ModularInference, self).__init__()
        self.leg_policy = copy.deepcopy(leg_policy)
        self.arm_policy = copy.deepcopy(arm_policy)

    def forward(self, observations):
        leg_action = self.leg_policy.act_inference(observations)
        arm_action = self.arm_policy.act_inference(observations)

        action = torch.cat((leg_action, arm_action), dim = -1)
        return action

    def export_onnx(self, path, filename):
        module = OnnxExporter(self.leg_policy, self.arm_policy)
        os.makedirs(path, exist_ok=True)
        full_path = osp.join(path, filename)
        module.export(full_path)
