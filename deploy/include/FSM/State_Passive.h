// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "unitree_articulation.h"
#include "isaaclab/assets/articulation/articulation.h"

#include <fstream>

#ifdef DEBUGSTREAM
bool readTxtTo2DVector(const std::string& file_path, std::vector<std::vector<float>>& data_2dvec) {
    // 清空二维vector
    data_2dvec.clear();
    std::ifstream in_file(file_path);
    if (!in_file.is_open()) {
        std::cerr << "[Error] 文件打开失败！路径：" << file_path << std::endl;
        return false;
    }

    std::string line;
    // 逐行读取文件内容
    while (std::getline(in_file, line)) {
        // 跳过空行（避免文件中的空行生成空的子vector）
        if (line.empty()) {
            continue;
        }

        // 将当前行转为字符串流，用于逐词读取数值
        std::istringstream line_stream(line);
        std::vector<float> line_data;  // 存储当前行的数值
        float num;

        // 读取当前行的所有浮点数值
        while (line_stream >> num) {
            line_data.push_back(num);
        }

        // 非空行才加入二维vector（避免空行）
        if (!line_data.empty()) {
            data_2dvec.push_back(line_data);
        }
    }

    in_file.close();

    // 校验有效数据
    if (data_2dvec.empty()) {
        std::cerr << "[Error] 文件中无有效浮点数值！" << std::endl;
        return false;
    }

    std::cout << "[Success] 文件读取完成！共读取 " << data_2dvec.size() << " 行，每行维度："
              << data_2dvec[0].size() << std::endl;
    return true;
}
#endif

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

#ifdef DEBUGSTREAM
        robot_ = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
        robot_->update();
        {
            std::string update_command = "RB + up.on_pressed";
            unitree::common::dsl::Parser p(update_command);
            auto ast = p.Parse();
            auto func = unitree::common::dsl::Compile(*ast);
            joystick_checks_ = [func]()->bool{ return func(FSMState::lowstate->joystick); };
        }

        for (int cid = 0; cid < sizeof(control_data_motor_ids_) / sizeof(control_data_motor_ids_[0]); cid ++) {
            std::cout << " cid" << cid << "  " << motor2controlmap_[control_data_motor_ids_[cid]] << std::endl;
            assert (motor2controlmap_[control_data_motor_ids_[cid]] == cid);
        }
        readTxtTo2DVector("mujoco_state.txt", control_data_);
#endif
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


#ifdef DEBUGSTREAM
    void pre_run()
    {
        FSMState::pre_run();
        robot_->update();

        if (joystick_checks_()) {
            update_q = true;
            dump_data_type ++;
            dump_data_type = dump_data_type % 13;
            control_times = 0;
            std::cout << "dump_data_type: " << dump_data_type << std::endl;
        }

        // if (dump_data_type == 1) {
        //     std::cout << "R ang_vel: " << robot_->data.root_ang_vel_b.transpose() <<
        //                  "R gravity_b: " << robot_->data.projected_gravity_b.transpose() <<
        //                  "R quat_w: " << robot_->data.root_quat_w << std::endl;
        //     std::cout << "R q0: " << lowstate->msg_.motor_state()[0].q() << std::endl;
        // }
    }
#endif

    void run()
    {
#ifdef DEBUGSTREAM
        if (control_times >= control_data_.size() || !update_q) {
            for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
            }
        } else {

            std::stringstream state_stream;
            std::stringstream control_stream;

            state_stream << std::right << std::fixed << std::setprecision(4);
            control_stream << std::right << std::fixed << std::setprecision(4);

            state_stream << "Squat_state: ";
            control_stream << "Squat_control: ";

            std::vector<float>& _data = control_data_[control_times];
            for(int motor_id(0); motor_id < _data.size(); ++motor_id)
            {
                int control_id = motor2controlmap_[motor_id];
                lowcmd->msg_.motor_cmd()[motor_id].q() = _data[control_id];

                state_stream << std::setw(8) << lowstate->msg_.motor_state()[motor_id].q()<< " ";
                control_stream << std::setw(8) << _data[control_id]<< " ";
            }

            state_stream << std::endl;
            state_stream << control_stream.str() << std::endl;
            std::cout << state_stream.str();

            control_times ++;
        }
#else
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
           lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
#endif
        return;
    }

#ifdef DEBUGSTREAM
    bool update_q = false;
    int dump_data_type = -1;
    std::function<bool()> joystick_checks_;
    std::shared_ptr<isaaclab::Articulation> robot_;
    std::vector<std::vector<float>> control_data_;
    int control_data_motor_ids_[13] = {0,  // 0
                          6,  // 1
                          12, // 2
                          1,  // 3
                          7,  // 4
                          2,  // 5
                          8,  // 6
                          3,  // 7
                          9,  // 8
                          4,  // 9
                          10, // 10
                          5,  // 11
                          11}; // 12
    int motor2controlmap_[13] = {0,    // 0
                               3,    // 1
                               5,    // 2
                               7,    // 3
                               9,    // 4
                               11,   // 5
                               1,    // 6
                               4,    // 7
                               6,    // 8
                               8,    // 9
                              10,    // 10
                              12,    // 11
                               2,    // 12
                              };

    int control_times = 0;
#endif
};

REGISTER_FSM(State_Passive)
