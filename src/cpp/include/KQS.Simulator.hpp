#pragma once

#include "KQS.Complex.hpp"

    
extern "C" __declspec(dllexport) void __cdecl ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
);


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Run(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots);
