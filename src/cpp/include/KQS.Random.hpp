#pragma once

#include "KQS.Utils.hpp"

#include <vector>
#include <array>
#include <span>


struct AliasTable {
    std::vector<double> Probs; // TODO align to 32 bytes
    std::vector<uint32> Aliases; // TODO align to 32 bytes
};


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(const std::vector<double> &probs);

template <ExecutionPolicy Policy>
std::vector<uint32>
SampleAliasTable(const AliasTable &table, const std::vector<uint32> &bins, const std::vector<double> &rands);


template <std::random_access_iterator Iterator>
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, Iterator out);

template <std::random_access_iterator Iterator>
void
GeneratePhilox8x4x32_10(const uint64 key, const std::array<uint64, 8> &counters, Iterator out);


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
