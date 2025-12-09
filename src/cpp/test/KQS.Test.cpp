#include <iostream>
#include <span>
#include <vector>
#include <execution>
#include <immintrin.h>
#include <ranges>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


#ifndef EXECUTION_POLICY
#define EXECUTION_POLICY Sequential
#endif

constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;
constexpr PrngAlgorithm Algorithm = PrngAlgorithm::Philox;


const std::map<std::string, std::vector<ExecutionPolicy>> FunctionExecutionMap = {
    { "_DeinterleaveAoSLComplex", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel } },
    { "_CalculateProbabilities", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel, ExecutionPolicy::Accelerated } },
    { "_Scale", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel } },
    { "GenerateRandomDiscrete", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel, ExecutionPolicy::Accelerated } },
    { "GenerateRandomContinuous", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel, ExecutionPolicy::Accelerated } },
    { "_SampleAliasTable", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel, ExecutionPolicy::Accelerated } },
    { "FlushSamples", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel } }
};


ExecutionPolicy GetFunctionExecutionPolicy(const std::string &name) {
    const auto it = FunctionExecutionMap.find(name);

    const auto &policies = it->second;
    for (const auto p : policies) {
        if (p == Policy) {
            return Policy;
        }
    }
    return policies.back();
}


std::string ExecutionPolicyToString(ExecutionPolicy policy) {
    switch (policy) {
        case ExecutionPolicy::Sequential:
            return "Sequential";
        case ExecutionPolicy::Parallel:
            return "Parallel";
        case ExecutionPolicy::Accelerated:
            return "Accelerated";
        default:
            return "Unknown";
    }
}


void PrintBenchmarkResult(const std::string &name) {
    const auto ToMillis = [] (double ns) {
        return ns / 1'000'000.0;
    };

    const auto result = BenchmarkRegistry::Instance().GetResult(name);
    std::cout << "* (" << ExecutionPolicyToString(GetFunctionExecutionPolicy(name)) << ") " << name << std::endl
              << "--- [Mean +- CI95: " << ToMillis(result.Mean) << " +- " << ToMillis(result.CI95) << " ms, \tMin: " << ToMillis(result.Min) << " ms, \tMax: " << ToMillis(result.Max) << " ms]" << std::endl;
}


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Test(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots) {

    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    PrintBenchmarkResult("_DeinterleaveAoSLComplex");
    
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    PrintBenchmarkResult("_CalculateProbabilities");
    
    const auto table = BuildAliasTable<Policy>(probs);
    PrintBenchmarkResult("_Scale");
    
    auto samples = SampleAliasTable<Policy, Algorithm>(table, NumShots);
    PrintBenchmarkResult("GenerateRandomDiscrete");
    PrintBenchmarkResult("GenerateRandomContinuous");
    PrintBenchmarkResult("_SampleAliasTable");
    
    FlushSamples<Policy>(StateCounts, samples);
    PrintBenchmarkResult("FlushSamples");
}


std::vector<LComplex> Generate1in1024UniformStateAmplitudes(size_t qubits) {
    const size_t numStates = 1ul << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);
    const double amplitude = 1.0 / std::sqrt(static_cast<double>(numStates) / 1024);
    for (size_t i = 0; i < numStates; ++i) {
        if (i % 1024 == 0) {
            stateAmplitudes[i] = { amplitude, 0.0 };
        }
    }
    return stateAmplitudes;
}

int main() {
    auto stateAmplitudes = Generate1in1024UniformStateAmplitudes(26); // 26 qubits -> 64Mi states -> 1GiB state vector
    std::vector<uint> stateCounts(stateAmplitudes.size(), 0);
    const uint numShots = 1024*1024*256; // 256Mi shots -> 1GiB samples
    Test<Policy, Algorithm>(stateCounts, stateAmplitudes, numShots);

    return 0;
}
