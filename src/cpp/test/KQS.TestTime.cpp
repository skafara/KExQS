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


template <PrngAlgorithm Algorithm>
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


void
TestRange() {
    BenchmarkRegistry &registry = BenchmarkRegistry::Instance();
    registry.Clear();

    const auto FlushProbabilityResult = [] (std::ofstream &fout, size_t qubits, const BenchmarkRegistry::Result &result) {
        const size_t NumStates = 1ul << qubits;
        fout << qubits << "\t" << NumStates << "\t"
             << result.Mean << "\t" << result.CI95 << "\t"
             << result.Min << "\t" << result.Max << "\n";        
    };

    const auto rangeQubits = std::views::iota(size_t{1}, size_t{27});
    {
        std::ofstream fout(DirResults + "/KQS.TestTime._CalculateProbabilities." + ExecutionPolicyToString(Policy) + ".txt");
        fout << "Qubits\tNumStates\tMean_ns\tCI95_ns\tMin_ns\tMax_ns\n";
        for (const size_t qubits : rangeQubits) {
            const auto StateAmplitudes = Generate1in1024UniformStateAmplitudes(qubits);
            const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
            const auto probs = CalculateProbabilities<Policy>(res, ims);
            FlushProbabilityResult(fout, qubits, registry.GetResult("_CalculateProbabilities"));
            registry.Clear();
        }
    }

    const auto StateAmplitudes = Generate1in1024UniformStateAmplitudes(4);
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    const auto table = BuildAliasTable<Policy>(probs);

    const auto FlushSamplingResult = [] (std::ofstream &fout, uint LogNumShots, const BenchmarkRegistry::Result &result) {
        const size_t NumShots = 1ul << LogNumShots;
        fout << LogNumShots << "\t" << NumShots << "\t"
             << result.Mean << "\t" << result.CI95 << "\t"
             << result.Min << "\t" << result.Max << "\n";        
    };
    
    {
        std::ofstream foutGenDiscrete(DirResults + "/KQS.GenerateRandomDiscrete." + ExecutionPolicyToString(Policy) + ".txt");
        std::ofstream foutGenContinuous(DirResults + "/KQS.GenerateRandomContinuous." + ExecutionPolicyToString(Policy) + ".txt");
        std::ofstream foutSampleAlias(DirResults + "/KQS._SampleAliasTable." + ExecutionPolicyToString(Policy) + ".txt");
        foutGenDiscrete << "LogNumShots\tNumShots\tMean_ns\tCI95_ns\tMin_ns\tMax_ns\n";
        foutGenContinuous << "LogNumShots\tNumShots\tMean_ns\tCI95_ns\tMin_ns\tMax_ns\n";
        foutSampleAlias << "LogNumShots\tNumShots\tMean_ns\tCI95_ns\tMin_ns\tMax_ns\n";

        const auto rangeLogShots = std::views::iota(uint{5}, uint{29});
        for (const uint LogNumShots : rangeLogShots) {
            const auto NumShots = 1u << LogNumShots;
            auto samples = SampleAliasTable<Policy, PrngAlgorithm::Philox>(table, NumShots);
            FlushSamplingResult(foutGenDiscrete, LogNumShots, registry.GetResult("GenerateRandomDiscrete"));
            FlushSamplingResult(foutGenContinuous, LogNumShots, registry.GetResult("GenerateRandomContinuous"));
            FlushSamplingResult(foutSampleAlias, LogNumShots, registry.GetResult("_SampleAliasTable"));
            registry.Clear();
        }
    }
}


int main() {
    // 26 qubits -> 64Mi states -> 1GiB state vector
    // 256Mi shots -> 1GiB samples
    const auto stateAmplitudes = Generate1in1024UniformStateAmplitudes(26);
    std::vector<uint> stateCounts(stateAmplitudes.size(), 0);
    const uint numShots = 1024*1024*256;
    Test<PrngAlgorithm::Philox>(stateCounts, stateAmplitudes, numShots);
    
    TestRange();
    return 0;
}
