#include "KQS.Complex.hpp"


using uint = unsigned int;

    
extern "C" __declspec(dllexport) void __cdecl ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumQubits,
    uint ANumStates,
    uint ANumShots
);
