
#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <math.h>
#include <cmath>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(squat_commands)
{
    const int left_knee_pitch_id = 3;
    const int right_knee_pitch_id = 9;
    const pi = ;
    std::vector<float> obs(7);
    auto & joystick = env->robot->data.joystick;
    const auto cfg = env->cfg["commands"]["suqat_command"]["ranges"];
    float times = (cfg["full_times"][0].as<float>() + cfg["full_times"][1].as<float>()) / 2;
    float phase_vel = M_PI / times;

    float suqat_command_phase = std::clamp(joystick->ly(), cfg["suqat_phase"][0].as<float>(), cfg["suqat_phase"][1].as<float>());

    float pos = (env->robot->data.joint_pos[left_knee_pitch_id] + env->robot->data.joint_pos[right_knee_pitch_id]) / 2
    float pos_phase = asin((env->cfg["commands"]["suqat_command"]["cpos"] - pos) / env->cfg["commands"]["suqat_command"]["rad"])

    if (pos_phase > suqat_command_phase) {
        phase_vel = - phase_vel;
        pos_phase += phase_vel * env->step_dt;
        if (pos_phase < suqat_command_phase)
            pos_phase = suqat_command_phase;
    } else {
        pos_phase += phase_vel * env->step_dt;
        if (pos_phase > suqat_command_phase)
            pos_phase = suqat_command_phase;
    }
    obs[0] = suqat_command_phase / M_PI;
    obs[1] = pos_phase / M_PI;

    obs[2] = cos(suqat_command_phase);
    obs[3] = cos(pos_phase);

    obs[4] = sin(suqat_command_phase);
    obs[5] = sin(pos_phase);

    obs[6] = phase_vel;

    return obs;
}

}
}
