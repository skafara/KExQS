#include <span>
#include <vector>
#include <complex>
#include <bit>
#include <random>

#include "KQS.Simulator.hpp"


static std::mt19937 gen(std::random_device{}());


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    const std::span<const LComplex> amplitudes(AStateAmplitudes, ANumStates);

    std::vector<std::complex<double>> amps(amplitudes.size());
    for (size_t i = 0; i < amps.size(); i++) {
        amps[i] = std::bit_cast<std::complex<double>>(amplitudes[i]);
    }

    std::vector<double> probs(amplitudes.size());
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] = std::norm(amps[i]);
    }
    
    std::discrete_distribution<uint> dist(probs.begin(), probs.end());
    for (size_t i = 0; i < ANumShots; i++) {
        const uint state = dist(gen);
        AStateCounts[state]++;
    }
}
