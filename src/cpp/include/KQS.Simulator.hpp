#pragma once

#include "KQS.Complex.hpp"


/**
 * @brief [EXTERNAL] Runs the quantum state measurement simulation.
 * @param AStateCounts Pointer to an array where the counts of each state will be stored.
 * @param AStateAmplitudes Pointer to an array of complex amplitudes representing the quantum state.
 * @param ANumStates The number of quantum states (size of the state vector).
 * @param ANumShots The number of measurement shots to simulate.
 * @return void
 */
extern "C" __declspec(dllexport) void __cdecl ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
);


/**
 * @brief Internal function to run the simulation with specified execution policy and PRNG algorithm.
 * @param StateCounts Span to store the counts of each state.
 * @param StateAmplitudes Span of complex amplitudes representing the quantum state.
 * @param NumShots The number of measurement shots to simulate.
 * @return void
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Run(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots);
