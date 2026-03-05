
#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <math.h>
#include <cmath>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(stomp_commands)
{
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["stomp_command"];
    float period = cfg["period"].as<float>();
    float offset0 = cfg["offset"][0].as<float>();
    float offset1 = cfg["offset"][1].as<float>();
    float threshold = cfg["threshold"].as<float>();

    bool stomp_flag = (joystick->ly() > 0.5) || (joystick->lx() > 0.5) || (joystick->rx() > 0.5);

    float global_phase = ((env->episode_length * env->step_dt) / period);

    float phase0 = std::fmod(global_phase + offset0, 1.0f);
    float phase1 = std::fmod(global_phase + offset1, 1.0f);

    float swing_phase0 = std::max(phase0 - threshold, 0.0f) / (1 - threshold);
    float swing_phase1 = std::max(phase1 - threshold, 0.0f) / (1 - threshold);

    if (! stomp_flag) {
        env->episode_length = 0;
        phase0 = 0;
        phase1 = 0;
        swing_phase0 = 0;
        swing_phase1 = 0;
    }

    std::vector<float> obs(8);

    obs[0] = std::sin(phase0 * M_PI * 2.0);
    obs[1] = std::sin(phase1 * M_PI * 2.0);

    obs[2] = std::cos(phase0 * M_PI * 2.0);
    obs[3] = std::cos(phase1 * M_PI * 2.0);

    obs[4] = std::sin(swing_phase0 * M_PI);
    obs[5] = std::sin(swing_phase1 * M_PI);
    obs[6] = std::cos(swing_phase0 * M_PI);
    obs[7] = std::cos(swing_phase1 * M_PI);

    return obs;
}

}
}
