#include <span>
#include <vector>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


#ifndef EXECUTION_POLICY
#define EXECUTION_POLICY Accelerated
#endif

#ifndef PRNG_ALGORITHM
#define PRNG_ALGORITHM Philox
#endif


constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;
constexpr PrngAlgorithm Algorithm = PrngAlgorithm::PRNG_ALGORITHM;


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Run(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    const auto table = BuildAliasTable<Policy>(probs);
    auto samples = SampleAliasTable<Policy, Algorithm>(table, NumShots);
    FlushSamples<Policy>(StateCounts, samples);
}


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    const std::span<const LComplex> StateAmplitudes(AStateAmplitudes, ANumStates);
    const std::span<uint> StateCounts(AStateCounts, ANumStates);
    
    if (Policy == ExecutionPolicy::Accelerated) {
        CLManager::Instance();
    }
    Run<Policy, Algorithm>(StateCounts, StateAmplitudes, ANumShots);
}
