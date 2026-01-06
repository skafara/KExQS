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


/** Simulator execution policy */
constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;
/** Simulator PRNG algorithm */
constexpr PrngAlgorithm Algorithm = PrngAlgorithm::PRNG_ALGORITHM;


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Run(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    // Deinterleave complex amplitudes into real and imaginary parts
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    // Calculate probabilities from amplitudes
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    // Build alias table
    const auto table = BuildAliasTable<Policy>(probs);
    // Sample from alias table
    auto samples = SampleAliasTable<Policy, Algorithm>(table, NumShots);
    // Flush samples into state counts
    FlushSamples<Policy>(StateCounts, samples);
}


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    // Wrap raw pointers into spans
    const std::span<const LComplex> StateAmplitudes(AStateAmplitudes, ANumStates);
    const std::span<uint> StateCounts(AStateCounts, ANumStates);
    
    // Initialize CLManager if using accelerated execution policy
    if (Policy == ExecutionPolicy::Accelerated) {
        CLManager::Instance();
    }
    // Run the simulation
    Run<Policy, Algorithm>(StateCounts, StateAmplitudes, ANumShots);
}
