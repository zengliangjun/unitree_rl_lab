#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <math.h>
#include <cmath>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(holosoma_phase_commands)
{
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["holosoma_phase_command"];
    float period = cfg["period"].as<float>();
    float offset0 = cfg["phase_offset"][0].as<float>();
    float offset1 = cfg["phase_offset"][1].as<float>();
    float stand_phase = cfg["stand_phase"].as<float>();

    float gait_freq = 1.0 / period;
    float phase_dt = 2 * M_PI * env->step_dt * gait_freq;

    float phase0 = std::fmod(env->episode_length * phase_dt + offset0 + M_PI, 2 * M_PI) - M_PI;
    float phase1 = std::fmod(env->episode_length * phase_dt + offset1 + M_PI, 2 * M_PI) - M_PI;


    bool velocity_flag = (abs(joystick->ly()) + abs(joystick->lx()) + abs(joystick->rx())) > 0.01;
    if (!velocity_flag) {
        phase0 = stand_phase;
        phase1 = stand_phase;
    }

    std::vector<float> obs(4);

    obs[0] = std::sin(phase0);
    obs[1] = std::sin(phase1);

    obs[2] = std::cos(phase0);
    obs[3] = std::cos(phase1);

    return obs;
}

}
}
