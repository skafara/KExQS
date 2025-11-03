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


inline
double
CalculateProbability(const double re, const double im);

inline
__m256d
CalculateProbability(const __m256d re, const __m256d im);


template <ExecutionPolicy Policy>
std::vector<double>
CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims);

template <ExecutionPolicy Policy>
void
_CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs);


template <ExecutionPolicy Policy>
void
FlushSamples(std::span<uint> StateCounts, const std::vector<uint> &samples);
