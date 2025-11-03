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
SampleAliasTable(const AliasTable &table, const uint ANumShots);

template <ExecutionPolicy Policy>
void
_SampleAliasTable(const AliasTable &table, std::span<const uint32> bins, std::span<const double> rands, std::span<uint32> samples);


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
void
_GenerateRandomContinuous(std::span<const uint64> u64_numbers, std::span<double> numbers);


template <ExecutionPolicy Policy>
std::vector<uint32>
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max);


template <ExecutionPolicy Policy>
void
_GenerateRandomDiscrete(std::span<const uint32> u32_numbers, const uint32 max, std::span<uint32> numbers);
