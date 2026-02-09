// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>

#ifdef DEBUGSTREAM
#include <fstream>
#include <chrono>
#endif

inline boost::bimap<int, std::string> FSMStringMap;

class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state)
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {
#ifdef DEBUGSTREAM
        _init_stream();
        std::cout << " _init_stream " << getStateString() << std::endl;
#endif
    }

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {

#ifdef DEBUGSTREAM
        if (debug_stream_.is_open())
            debug_stream_.close();
        std::cout << " _close_stream " << getStateString() << std::endl;
#endif
    }

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;

#ifdef DEBUGSTREAM
    std::string _generate_time_filename() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        char time_buf[20];
        std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", std::localtime(&now_c));

        std::string state = getStateString();
        return state + "_" + std::string(time_buf) + ".log";
    }

    void _init_stream() {
        if (debug_stream_.is_open())
            return;

        std::string fname = _generate_time_filename();
        debug_stream_ = std::fstream(fname, std::ios::out);
        std::cout << " open fname " << fname << std::endl;
    }

    std::fstream debug_stream_;
#endif

private:
    int state_;
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
            std::cout << " register: " << #Derived << std::endl;                        \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \

