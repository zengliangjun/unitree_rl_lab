// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "unitree_articulation.h"
#include "isaaclab/assets/articulation/articulation.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive")
    : FSMState(state, state_string)
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }

        robot_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

        robot_->update();
        {
            std::string update_command = "RB + up.on_pressed";
            unitree::common::dsl::Parser p(update_command);
            auto ast = p.Parse();
            auto func = unitree::common::dsl::Compile(*ast);
            joystick_checks_ = [func]()->bool{ return func(FSMState::lowstate->joystick); };
        }
    }

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    void pre_run()
    {
        FSMState::pre_run();
        robot_->update();
        if (joystick_checks_()) {
            update_q = true;
            dump_data_type ++;
            dump_data_type = dump_data_type % 39;
            std::cout << "dump_data_type: " << dump_data_type << std::endl;
        }

        // if (dump_data_type == 1) {
        //     std::cout << "R ang_vel: " << robot_->data.root_ang_vel_b.transpose() <<
        //                  "R gravity_b: " << robot_->data.projected_gravity_b.transpose() <<
        //                  "R quat_w: " << robot_->data.root_quat_w << std::endl;
        //     std::cout << "R q0: " << lowstate->msg_.motor_state()[0].q() << std::endl;
        // }
    }

    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            int s = dump_data_type % 13;
            if (update_q && s == i) {

                int i_mod = dump_data_type / 13;

                if (i_mod == 0) {
                    lowcmd->msg_.motor_cmd()[i].q() = 0.1;
                } else if (i_mod == 1) {
                    lowcmd->msg_.motor_cmd()[i].q() = 0.1;
                } else if (i_mod == 2) {
                    lowcmd->msg_.motor_cmd()[i].q() = 0.1;
                }
                std::cout << "Set motor: " << i << " to q: " << lowcmd->msg_.motor_cmd()[i].q() << std::endl;
            }
            // if (update_q && 9 == i) {
            //     if (dump_data_type % 2 == 0) {
            //         lowcmd->msg_.motor_cmd()[i].q() = 0.2;
            //     } else {
            //         lowcmd->msg_.motor_cmd()[i].q() = 0;
            //     }
            //     std::cout << "Set motor: " << i << " to q: " << lowcmd->msg_.motor_cmd()[i].q() << std::endl;
            // }
            else {
                lowcmd->msg_.motor_cmd()[i].q() = 0; // lowstate->msg_.motor_state()[i].q();
            }
        }
        // if (update_q) {
        //     update_q = false;
        // }
    }

    bool update_q = false;
    int dump_data_type = -1;
    std::function<bool()> joystick_checks_;
    std::shared_ptr<isaaclab::Articulation> robot_;
};

REGISTER_FSM(State_Passive)
