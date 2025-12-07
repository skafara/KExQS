#pragma once

#include "KQS.Utils.hpp"

#include <vector>
#include <array>
#include <span>
#include <ranges>


struct AliasTable {
    std::vector<double> Probs; // TODO align to 64 bytes
    std::vector<uint32> Aliases; // TODO align to 64 bytes
};


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(const std::vector<double> &probs);

template <ExecutionPolicy Policy>
void
_Scale(const std::vector<double> &probs, std::vector<double> &scaled);

template <ExecutionPolicy Policy>
std::vector<uint32>
SampleAliasTable(const AliasTable &table, const uint NumShots);

template <ExecutionPolicy Policy>
inline
void
_SampleAliasTable(const AliasTable &table, const typename DeviceContainer<Policy, uint32>::type &bins, const typename DeviceContainer<Policy, double>::type &rands, std::span<uint32> samples);


template <std::random_access_iterator Iterator>
inline
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, Iterator out);

template <std::random_access_iterator Iterator, std::ranges::input_range Range>
requires std::same_as<std::ranges::range_value_t<Range>, uint64>
inline
void
GeneratePhilox8x4x32_10(const uint64 key, Range counters, Iterator out);


template <ExecutionPolicy Policy>
std::vector<uint32>
GenerateRandomUint32(const uint64 key, const size_t count);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint32(const uint64 key, const size_t count, std::span<uint32> numbers);


template <ExecutionPolicy Policy>
std::vector<uint64>
GenerateRandomUint64(const uint64 key, const size_t count);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint64(const uint64 key, const size_t count, std::span<uint64> numbers);


template <ExecutionPolicy Policy>
DeviceContainer<Policy, double>::type
GenerateRandomContinuous(const uint64 key, const size_t count);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomContinuous(std::span<const uint64> u64_numbers, std::span<double> numbers);


template <ExecutionPolicy Policy>
DeviceContainer<Policy, uint32>::type
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max);


template <ExecutionPolicy Policy>
inline
void
_GenerateRandomDiscrete(std::span<const uint32> u32_numbers, const uint32 max, std::span<uint32> numbers);
