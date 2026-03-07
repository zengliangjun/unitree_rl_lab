#include "State_Velocity2.h"

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_Velocity2::State_Velocity2(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Velocity2::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

#ifdef DEBUGSTREAM
    std::stringstream state_stream;
    std::stringstream control_stream;

    state_stream << std::right << std::fixed << std::setprecision(4);
    control_stream << std::right << std::fixed << std::setprecision(4);

    state_stream << "RLBase_state: ";
    control_stream << "RLBase_control: ";
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {

        state_stream << std::setw(8) << lowstate->msg_.motor_state()[i].q()<< " ";
        control_stream << std::setw(8) << lowcmd->msg_.motor_cmd()[i].q() << " ";
    }
    state_stream << std::endl;
    state_stream << control_stream.str() << std::endl;
    debug_stream_ << state_stream.str();

    debug_stream_ << "ang_vel: " << env->robot->data.root_ang_vel_b.transpose() << std::endl <<
                     "gravity_b: " << env->robot->data.projected_gravity_b.transpose() << std::endl <<
                     "quat_w: " << env->robot->data.root_quat_w.coeffs().transpose() << std::endl;

#endif

}
