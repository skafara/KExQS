#pragma once

#include "KQS.Utils.hpp"

#include <vector>
#include <array>
#include <span>


using uint32 = uint32_t;
using uint64 = uint64_t;


struct AliasTable {
    std::vector<double> Probs; // TODO align to 32 bytes
    std::vector<size_t> Aliases; // TODO align to 32 bytes
};


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(const std::vector<double> &probs);


template <ExecutionPolicy Policy>
std::vector<uint32_t>
SampleAliasTable(const AliasTable &table, const std::vector<uint32_t> &bins, const std::vector<double> &rands);


std::array<uint32, 4>
GeneratePhilox4x32_10(const uint64 key, const uint64 counter);


std::array<uint32, 32>
GeneratePhilox8x4x32_10(const uint64 key, const std::span<uint64> counters);


template <ExecutionPolicy Policy>
std::vector<uint32>
GenerateRandomUint32(const uint64 key, const size_t count);


template <ExecutionPolicy Policy>
std::vector<uint64>
GenerateRandomUint64(const uint64 key, const size_t count);


template <ExecutionPolicy Policy>
std::vector<double>
GenerateRandomContinuous(const uint64 key, const size_t count);


template <ExecutionPolicy Policy>
std::vector<uint32>
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max);
