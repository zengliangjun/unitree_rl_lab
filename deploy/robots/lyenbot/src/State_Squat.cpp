#include "State_Squat.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/squat_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_Squat::State_Squat(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    /** State_RLBase  */
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

    /** State_RLBase  */
    {
        std::string update_command = env->cfg["commands"]["squat_command"]["update_command"].as<std::string>(); // "RT + Y.on_pressed";
        unitree::common::dsl::Parser p(update_command);
        auto ast = p.Parse();
        auto func = unitree::common::dsl::Compile(*ast);
        squat_command_checks = [func]()->bool{ return func(FSMState::lowstate->joystick); };
    }

    {
        std::string update_time = env->cfg["commands"]["squat_command"]["update_time"].as<std::string>(); // "RT + X.on_pressed";
        unitree::common::dsl::Parser p(update_time);
        auto ast = p.Parse();
        auto func = unitree::common::dsl::Compile(*ast);
        update_time_checks = [func]()->bool{ return func(FSMState::lowstate->joystick); };
    }

    // for squat
    env->robot->data.left_knee_id = env->cfg["commands"]["squat_command"]["left_knee_id"].as<int>(); // 3;
    env->robot->data.right_knee_id = env->cfg["commands"]["squat_command"]["right_knee_id"].as<int>(); // 9;

    // _resample_init_compute();
}

void State_Squat::_resample_init_compute() {

    int left_knee_id = env->robot->data.left_knee_id;
    int right_knee_id = env->robot->data.right_knee_id;

    // command
    const auto cfg = env->cfg["commands"]["squat_command"]["ranges"];
    float squat_command_phase = cfg["squat_phase"][1].as<float>();
    env->robot->data.squat_command_phase = squat_command_phase;


    // phase_vel
    float time = (cfg["full_times"][0].as<float>() + cfg["full_times"][1].as<float>()) / 2;
    float phase_vel = M_PI / time;
    env->robot->data.full_time = time;

    // phase
    float pos = (env->robot->data.joint_pos[left_knee_id] + \
        env->robot->data.joint_pos[right_knee_id]) / 2;

    float sin = (env->cfg["commands"]["squat_command"]["cpos"].as<float>() - pos) / \
        env->cfg["commands"]["squat_command"]["rad"].as<float>();

    sin = std::clamp(sin, -1.0f, 1.0f);
    float pos_phase = asin(sin);
    env->robot->data.pos_phase = pos_phase;


    std::cout << "init: command_phase: cpos " << env->cfg["commands"]["squat_command"]["cpos"].as<float>() \
                                         << " rad:  " << env->cfg["commands"]["squat_command"]["rad"].as<float>() \
                                         << " left:  " << left_knee_id  \
                                         << " right:  " << right_knee_id  \
                                         << " pos:  " << pos << std::endl;

    std::cout << "joint_pos:  " << env->robot->data.joint_pos << std::endl;

    if (pos_phase > squat_command_phase) {
        phase_vel = - phase_vel;
    }
    env->robot->data.phase_vel = phase_vel;

    std::cout << "init: command_phase: " << squat_command_phase << " pos_phase:  " << pos_phase << "  phase_vel:  " << phase_vel << std::endl;
}

void State_Squat::_resample_compute() {
    int left_knee_id = env->robot->data.left_knee_id;
    int right_knee_id = env->robot->data.right_knee_id;

    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["squat_command"]["ranges"]["squat_phase"];

    // command
    float squat_command_phase = joystick->ly() * M_PI / 2;
    squat_command_phase = std::clamp(squat_command_phase, cfg[0].as<float>(), cfg[1].as<float>());
    env->robot->data.squat_command_phase = squat_command_phase;

    // phase_vel
    // float times = (cfg["full_times"][0].as<float>() + cfg["full_times"][1].as<float>()) / 2;
    float phase_vel = M_PI / env->robot->data.full_time;

    // phase
    float pos = (env->robot->data.joint_pos[left_knee_id] + \
        env->robot->data.joint_pos[right_knee_id]) / 2;

    float sin = (env->cfg["commands"]["squat_command"]["cpos"].as<float>() - pos) / \
        env->cfg["commands"]["squat_command"]["rad"].as<float>();

    sin = std::clamp(sin, -1.0f, 1.0f);
    float pos_phase = asin(sin);
    env->robot->data.pos_phase = pos_phase;

    if (pos_phase > squat_command_phase) {
        phase_vel = - phase_vel;
    }
    env->robot->data.phase_vel = phase_vel;
    std::cout << "resample: command_phase: " << squat_command_phase << " pos_phase:  " << pos_phase << "  phase_vel:  " << phase_vel << std::endl;
}

void State_Squat::pre_run()
{
    FSMState::pre_run();

    if (squat_command_checks()) {
        _resample_compute();
    }
    if (update_time_checks()) {
        auto & joystick = env->robot->data.joystick;
        const auto cfg = env->cfg["commands"]["squat_command"]["ranges"]["full_times"];
        float min = cfg[0].as<float>();
        float max = cfg[1].as<float>();
        float time = (joystick->ly() + 1) / 2 * (max - min) + min;
        env->robot->data.full_time = time;
        std::cout << "update_time: " << time << std::endl;
    }
}

void State_Squat::run()
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
