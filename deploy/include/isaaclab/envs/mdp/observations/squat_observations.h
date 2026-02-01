
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
    auto & asset = env->robot;
    float squat_command_phase = asset->data.squat_command_phase;
    float pos_phase = asset->data.pos_phase;
    float phase_vel = asset->data.phase_vel;

    float pos_phase_new = pos_phase + phase_vel * env->step_dt;
    if (phase_vel > 0) {
        if (pos_phase_new > squat_command_phase)
            pos_phase_new = squat_command_phase;
    } else {
        if (pos_phase_new < squat_command_phase)
            pos_phase_new = squat_command_phase;
    }
    asset->data.pos_phase = pos_phase_new;

    std::vector<float> obs(7);
    obs[0] = squat_command_phase / M_PI;
    obs[1] = pos_phase_new / M_PI;

    obs[2] = cos(squat_command_phase);
    obs[3] = cos(pos_phase_new);

    obs[4] = sin(squat_command_phase);
    obs[5] = sin(pos_phase_new);

    obs[6] = phase_vel;

    // std::cout << "OBSERVATION: command_phase: " << squat_command_phase << " pos_phase:  " << pos_phase << "  phase_vel:  " << phase_vel << std::endl;
    return obs;
}

}
}
