#include <iostream>
#include <span>
#include <vector>
#include <execution>
#include <immintrin.h>
#include <ranges>
#include <fstream>
#include <random>
#include <filesystem>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


#ifndef EXECUTION_POLICY
#define EXECUTION_POLICY Accelerated
#endif


constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;

const std::string DirResults = "results";


inline
void
Test(std::span<uint> StateCountsRandomOrg, std::span<uint> StateCountsPhilox, std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);    
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    const auto table = BuildAliasTable<Policy>(probs);
    
    auto samplesRandomOrg = SampleAliasTable<Policy, PrngAlgorithm::RandomOrg>(table, NumShots);
    auto samplesPhilox = SampleAliasTable<Policy, PrngAlgorithm::Philox>(table, NumShots);
    
    FlushSamples<Policy>(StateCountsRandomOrg, samplesRandomOrg);
    FlushSamples<Policy>(StateCountsPhilox, samplesPhilox);
}


std::vector<LComplex> GenerateUniformStateVector(size_t qubits) {
    const size_t numStates = 1ul << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);
    const double amplitude = 1.0 / std::sqrt(static_cast<double>(numStates));
    for (size_t i = 0; i < numStates; ++i) {
        stateAmplitudes[i] = { amplitude, 0.0 };
    }

    return stateAmplitudes;
}

std::vector<LComplex> GenerateSpikyStateVector(size_t qubits) {
    const size_t numStates = 1ul << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);

    double pSpike = 0.5;
    int K = stateAmplitudes.size();

    stateAmplitudes[0].Re = std::sqrt(pSpike);
    stateAmplitudes[0].Im = 0.0;

    double pRest = (1.0 - pSpike) / (K - 1);
    double ampRest = std::sqrt(pRest);

    for (int i = 1; i < K; i++) {
        stateAmplitudes[i].Re = ampRest;
        stateAmplitudes[i].Im = 0.0;
    }

    return stateAmplitudes;
}

std::vector<LComplex> GenerateMultiSpikeStateVector(size_t qubits) {
    const size_t numStates = 1ull << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);

    const size_t numSpikes = 10;
    const double spikeTotalProb = 0.5;

    // Probability per spike
    const double pSpike = spikeTotalProb / numSpikes;
    const double ampSpike = std::sqrt(pSpike);

    // Probability per remaining non-spike bin
    const double pRest = (1.0 - spikeTotalProb) / (numStates - numSpikes);
    const double ampRest = std::sqrt(pRest);

    // Fill spikes at indices 0..numSpikes-1
    for (size_t i = 0; i < numSpikes; i++) {
        stateAmplitudes[i] = { ampSpike, 0.0 };
    }

    // Fill remaining indices
    for (size_t i = numSpikes; i < numStates; i++) {
        stateAmplitudes[i] = { ampRest, 0.0 };
    }

    return stateAmplitudes;
}

std::vector<LComplex> GenerateExponentialStateVector(size_t qubits) {
    const size_t numStates = 1ull << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);

    const double tau = 300.0;

    // First compute unnormalized probabilities
    std::vector<double> probs(numStates);
    double sum = 0.0;

    for (size_t i = 0; i < numStates; i++) {
        probs[i] = std::exp(-static_cast<double>(i) / tau);
        sum += probs[i];
    }

    // Normalize and convert to amplitudes
    for (size_t i = 0; i < numStates; i++) {
        double p = probs[i] / sum;
        stateAmplitudes[i] = { std::sqrt(p), 0.0 };
    }

    return stateAmplitudes;
}

std::vector<LComplex> GenerateDirichletStateVector(size_t qubits) {
    const size_t numStates = 1ull << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);

    const double alpha = 1.0;
    const uint32_t seed = 42;
    
    std::mt19937 gen(seed);
    std::gamma_distribution<double> gamma(alpha, 1.0);

    std::vector<double> values(numStates);
    double sum = 0.0;

    // Sample Gamma(alpha,1) and normalize → Dirichlet distribution
    for (size_t i = 0; i < numStates; i++) {
        values[i] = gamma(gen);
        sum += values[i];
    }

    for (size_t i = 0; i < numStates; i++) {
        double p = values[i] / sum;
        stateAmplitudes[i] = { std::sqrt(p), 0.0 };
    }

    return stateAmplitudes;
}


int main() {
    constexpr size_t qubits = 12; // 12 qubits -> 4K states
    constexpr uint NumShots = 1024 * 1024;  // 1M shots -> (~256 samples/state)

    const std::vector<std::pair<std::string,
        std::vector<LComplex>(*)(size_t)>> generators = {
        { "Uniform",      GenerateUniformStateVector },
        { "Spiky",        GenerateSpikyStateVector },
        { "MultiSpike",   GenerateMultiSpikeStateVector },
        { "Exponential",  GenerateExponentialStateVector },
        { "Dirichlet",    GenerateDirichletStateVector }
    };

    std::filesystem::create_directories(DirResults);
    for (const auto& [name, generator] : generators)
    {
        // 1. Generate state amplitudes
        auto StateAmplitudes = generator(qubits);

        // 2. Allocate counters
        std::vector<uint> StateCountsRandomOrg(StateAmplitudes.size(), 0);
        std::vector<uint> StateCountsPhilox(StateAmplitudes.size(), 0);

        // 3. Run sampling test
        Test(StateCountsRandomOrg, StateCountsPhilox, StateAmplitudes, NumShots);

        // 4. Write output
        const std::string fileTrue  = DirResults + "/KQS.TestDistribution." + name + ".RandomOrg.txt";
        const std::string filePRNG  = DirResults + "/KQS.TestDistribution." + name + ".Philox.txt";

        {
            std::ofstream fout(fileTrue);
            for (size_t i = 0; i < StateCountsRandomOrg.size(); ++i)
                fout << StateCountsRandomOrg[i] << "\n";
        }
        {
            std::ofstream fout(filePRNG);
            for (size_t i = 0; i < StateCountsPhilox.size(); ++i)
                fout << StateCountsPhilox[i] << "\n";
        }
    }
    return 0;
}
