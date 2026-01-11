#include "State_Squat.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/squat_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_Squat::State_Squat(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    /*
    const auto & joy = FSMState::lowstate->joystick;
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > env->robot->data.motion_loader->duration; }, // time out
            FSMStringMap.right.at("Velocity")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            FSMStringMap.right.at("Passive")
        )
    );
    */
}
