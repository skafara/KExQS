#include <iostream>
#include <span>
#include <vector>
#include <execution>
#include <immintrin.h>
#include <ranges>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <fstream>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


#ifndef EXECUTION_POLICY
#define EXECUTION_POLICY Sequential
#endif

constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;


const std::string DirResults = "results";


const std::map<std::string, std::vector<ExecutionPolicy>> FunctionExecutionMap = {
    { "WholeProcess", { ExecutionPolicy::Sequential, ExecutionPolicy::Parallel, ExecutionPolicy::Accelerated } }
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
            throw std::runtime_error("Unknown ExecutionPolicy");
    }
}


double ToMillis(double ns) {
    return ns / 1'000'000.0;
}


void PrintBenchmarkResult(const std::string &name) {
    const auto result = BenchmarkRegistry::Instance().GetResult(name);
    std::cout << "* (" << ExecutionPolicyToString(GetFunctionExecutionPolicy(name)) << ") " << name << std::endl
              << "--- [Mean +- CI95: " << ToMillis(result.Mean) << " +- " << ToMillis(result.CI95) << " ms, \tMin: " << ToMillis(result.Min) << " ms, \tMax: " << ToMillis(result.Max) << " ms]" << std::endl;
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


void
TestWhole(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    for (size_t i = 0; i < BenchmarkingWarmupIterations; ++i) {
        const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
        const auto probs = CalculateProbabilities<Policy>(res, ims);
        const auto table = BuildAliasTable<Policy>(probs);
        auto samples = SampleAliasTable<Policy, PrngAlgorithm::Philox>(table, NumShots);
        FlushSamples<Policy>(StateCounts, samples);
    }
    auto &registry = BenchmarkRegistry::Instance();
    registry.Clear();
    for (size_t i = 0; i < BenchmarkingMeasuredIterations; ++i) {
        ScopedTimer timer("WholeProcess");
        const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
        const auto probs = CalculateProbabilities<Policy>(res, ims);
        const auto table = BuildAliasTable<Policy>(probs);
        auto samples = SampleAliasTable<Policy, PrngAlgorithm::Philox>(table, NumShots);
        FlushSamples<Policy>(StateCounts, samples);
    }
    PrintBenchmarkResult("WholeProcess");
}


int main() {
    // 26 qubits -> 64Mi states -> 1GiB state vector
    // 256Mi shots -> 1GiB samples
    const auto stateAmplitudes = Generate1in1024UniformStateAmplitudes(26);
    std::vector<uint> stateCounts(stateAmplitudes.size(), 0);
    const uint numShots = 1024*1024*256;
    TestWhole(stateCounts, stateAmplitudes, numShots);
    return 0;
}
