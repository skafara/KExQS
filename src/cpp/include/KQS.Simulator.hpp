#pragma once

#include "KQS.Complex.hpp"

    
extern "C" __declspec(dllexport) void __cdecl ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
);


template <ExecutionPolicy Policy>
void
Run(const std::span<uint> StateCounts, const std::span<const LComplex> StateAmplitudes, const uint NumShots);
