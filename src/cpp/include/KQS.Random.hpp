#pragma once

#include "KQS.Utils.hpp"

#include <vector>
#include <array>
#include <span>
#include <ranges>


struct AliasTable {
    AlignedVector64<double> Probs;
    AlignedVector64<uint32> Aliases;
};


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(std::span<const double> probs);

template <ExecutionPolicy Policy>
void
_Scale(std::span<const double> probs, std::span<double> scaled);


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
AlignedVector64<uint32>
SampleAliasTable(const AliasTable &table, const uint NumShots);

template <ExecutionPolicy Policy>
inline
void
_SampleAliasTable(const AliasTable &table, typename DeviceContainer<Policy, uint32>::const_ref_type bins, typename DeviceContainer<Policy, double>::const_ref_type rands, std::span<uint32> samples);


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
void
GenerateRandomUint32(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint32>::ref_type numbers);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint32(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint32>::ref_type numbers);


template <ExecutionPolicy Policy>
void
GenerateRandomUint64(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint64>::ref_type numbers);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint64(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint64>::ref_type numbers);


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, double>::type
GenerateRandomContinuous(const uint64 key, const size_t count);

template <ExecutionPolicy Policy>
inline
void
_GenerateRandomContinuous(typename DeviceContainer<Policy, uint64>::const_ref_type u64_numbers, typename DeviceContainer<Policy, double>::ref_type numbers);


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, uint32>::type
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max);


template <ExecutionPolicy Policy>
inline
void
_GenerateRandomDiscrete(typename DeviceContainer<Policy, uint32>::const_ref_type u32_numbers, const uint32 max, typename DeviceContainer<Policy, uint32>::ref_type numbers);
