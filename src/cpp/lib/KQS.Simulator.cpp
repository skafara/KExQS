#include <span>
#include <vector>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


constexpr ExecutionPolicy Policy = ExecutionPolicy::Accelerated;


template <ExecutionPolicy Policy>
void
Run(const std::span<uint> StateCounts, const std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    const auto table = BuildAliasTable<Policy>(probs);
    const auto samples = SampleAliasTable<Policy>(table, NumShots);
    FlushSamples<Policy>(StateCounts, samples);
}


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    const std::span<const LComplex> StateAmplitudes(AStateAmplitudes, ANumStates);
    std::span<uint> StateCounts(AStateCounts, ANumStates);
    
    if constexpr (Policy == ExecutionPolicy::Accelerated) {
        CLManager::Instance();
    }
    Run<Policy>(StateCounts, StateAmplitudes, ANumShots);
}
