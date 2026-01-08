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
auto
Test(std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    std::cout << "Deinterleaving state amplitudes..." << std::endl;
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);    
    std::cout << "Calculating probabilities..." << std::endl;
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    std::cout << "Building alias table..." << std::endl;
    const auto table = BuildAliasTable<Policy>(probs);
    
    std::cout << "Sampling using Random.org PRNG..." << std::endl;
    auto samplesRandomOrg = SampleAliasTable<Policy, PrngAlgorithm::RandomOrg>(table, NumShots);
    std::cout << "Sampling using Philox PRNG..." << std::endl;
    auto samplesPhilox = SampleAliasTable<Policy, PrngAlgorithm::Philox>(table, NumShots);

    return std::make_pair(samplesPhilox, samplesRandomOrg);
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

std::vector<LComplex> GenerateNormalStateVector(size_t qubits) {
    const size_t numStates = 1ull << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);

    const double mean = static_cast<double>(numStates) / 2.0;
    const double stddev = static_cast<double>(numStates) / 8.0;

    // First compute unnormalized probabilities
    std::vector<double> probs(numStates);
    double sum = 0.0;
    for (size_t i = 0; i < numStates; i++) {
        double diff = static_cast<double>(i) - mean;
        probs[i] = std::exp(-0.5 * (diff * diff) / (stddev * stddev));
        sum += probs[i];
    }

    // Normalize and convert to amplitudes
    for (size_t i = 0; i < numStates; i++) {
        double p = probs[i] / sum;
        stateAmplitudes[i] = { std::sqrt(p), 0.0 };
    }
    
    return stateAmplitudes;
}

static constexpr size_t BUF_SIZE = 8 * 1024 * 1024; // 8 MB buffer

void write_samples_fast(const std::string& filename, const AlignedVector64<uint32> &samples) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint64_t count = samples.size();
    fout.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Convert once to uint16_t buffer
    std::vector<uint16_t> buf;
    buf.reserve(samples.size());

    for (uint32 v : samples) {
        buf.push_back(static_cast<uint16_t>(v));
    }

    fout.write(
        reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(uint16_t)
    );

    if (!fout) {
        throw std::runtime_error("Error while writing file: " + filename);
    }
}


int main() {
    constexpr size_t qubits = 10; // 10 qubits -> 1K states
    constexpr uint NumShots = 32 * 1024 * 1024;  // 32M shots -> (~32K samples/state)

    const std::vector<std::pair<std::string,
        std::vector<LComplex>(*)(size_t)>> generators = {
        { "Uniform",      GenerateUniformStateVector },
        { "Spiky",        GenerateSpikyStateVector },
        { "MultiSpike",   GenerateMultiSpikeStateVector },
        { "Exponential",  GenerateExponentialStateVector },
        { "Normal",       GenerateNormalStateVector }
    };

    std::filesystem::create_directories(DirResults);
    for (const auto& [name, generator] : generators)
    {
        // 1. Generate state amplitudes
        std::cout << "Generating state amplitudes for " << name << " distribution..." << std::endl;
        const auto StateAmplitudes = generator(qubits);
        std::cout << "Done." << std::endl;

        // 2. Run sampling test
        const auto [samplesPhilox, samplesRandomOrg] = Test(StateAmplitudes, NumShots);

        // 3. Write output
        const std::string fileTrue  = DirResults + "/KQS.TestDistribution." + name + ".RandomOrg.txt";
        const std::string filePRNG  = DirResults + "/KQS.TestDistribution." + name + ".Philox.txt";

        {
            std::cout << "Writing results to " << fileTrue << "..." << std::endl;
            write_samples_fast(fileTrue, samplesRandomOrg);
        }
        {
            std::cout << "Writing results to " << filePRNG << "..." << std::endl;
            write_samples_fast(filePRNG, samplesPhilox);
        }
    }
    return 0;
}
