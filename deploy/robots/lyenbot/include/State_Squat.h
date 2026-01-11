#pragma once

#include "FSM/State_RLBase.h"

class State_Squat : public State_RLBase
{
public:
    State_Squat(int state_mode, std::string state_string);

};

REGISTER_FSM(State_Squat)