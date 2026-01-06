#pragma once

#include "KQS.Utils.hpp"

#include <vector>
#include <array>
#include <span>
#include <ranges>


/** @brief Seed for random discrete number generation */
constexpr uint64 SeedRandomDiscrete = 42ul;
/** @brief Seed for random continuous number generation */
constexpr uint64 SeedRandomContinuous = 43ul;


/**
 * @brief Alias table for efficient discrete sampling
 */
struct AliasTable {
    /** Probability table */
    AlignedVector64<double> Probs;
    /** Alias table */
    AlignedVector64<uint32> Aliases;
};


/**
 * @brief [WRAPPER] [Sequential] Build alias table from probability distribution
 * @param probs Probability distribution
 * @return Alias table
 */
template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(std::span<const double> probs);

/**
 * @brief [DELEGATE] Scale probabilities for alias method
 * @param probs Input probabilities
 * @param scaled Scaled probabilities
 * @return void
 */
template <ExecutionPolicy Policy>
void
_Scale(std::span<const double> probs, std::span<double> scaled);


/**
 * @brief [WRAPPER] Sample from alias table
 * @param table Alias table
 * @param NumShots Number of samples
 * @return Samples
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
AlignedVector64<uint32>
SampleAliasTable(const AliasTable &table, const uint NumShots);

/**
 * @brief [DELEGATE] Sample from alias table
 * @param table Alias table
 * @param bins Bin indices
 * @param rands Random numbers
 * @param samples Output samples
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_SampleAliasTable(const AliasTable &table, typename DeviceContainer<Policy, uint32>::ref_const_type bins, typename DeviceContainer<Policy, double>::ref_const_type rands, std::span<uint32> samples);


/**
 * @brief Generate Philox 4x32-10round random numbers
 * @param key Key
 * @param counter Counter
 * @param out Output iterator to write into the generated numbers
 * @return void
 */
template <std::random_access_iterator Iterator>
inline
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, Iterator out);

/**
 * @brief Generate Philox 8x 4x32-10round random numbers
 * @param key Key
 * @param counters Range (8 elements) of counters
 * @param out Output iterator to write into the generated numbers
 * @return void
 */
template <std::random_access_iterator Iterator, std::ranges::input_range Range>
requires std::same_as<std::ranges::range_value_t<Range>, uint64>
inline
void
GeneratePhilox8x4x32_10(const uint64 key, Range counters, Iterator out);


/**
 * @brief [DELEGATE] Generate random uint32 numbers
 * @param key Key
 * @param count Number of random numbers to generate
 * @param numbers Output container for random numbers
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint32(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint32>::ref_type numbers);


/**
 * @brief Generate random uint64 numbers
 * @param key Key
 * @param count Number of random numbers to generate
 * @param numbers Output container for random numbers
 * @return void
 */
template <ExecutionPolicy Policy>
void
GenerateRandomUint64(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint64>::ref_type numbers);

/**
 * @brief [DELEGATE] Generate random uint64 numbers
 * @param key Key
 * @param count Number of random numbers to generate
 * @param numbers Output container for random numbers
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_GenerateRandomUint64(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint64>::ref_type numbers);


/**
 * @brief [WRAPPER] Generate random continuous numbers in [0, 1)
 * @param key Key
 * @param count Number of random numbers to generate
 * @return Container with generated random numbers
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, double>::type
GenerateRandomContinuous(const uint64 key, const size_t count);

/**
 * @brief [DELEGATE] Generate random continuous numbers in [0, 1)
 * @param key Key
 * @param count Number of random numbers to generate
 * @param numbers Output container for random numbers
 * @return void
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
_GenerateRandomContinuous(const uint64 key, const size_t count, typename DeviceContainer<Policy, double>::ref_type numbers);


/**
 * @brief [DELEGATE] Transform uint64 random numbers to continuous [0, 1)
 * @param u64_numbers Input uint64 random numbers
 * @param numbers Output continuous random numbers
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_TransformContinuous(typename DeviceContainer<Policy, uint64>::ref_const_type u64_numbers, typename DeviceContainer<Policy, double>::ref_type numbers);


/**
 * @brief [WRAPPER] Generate random discrete numbers in [0, max)
 * @param key Key
 * @param count Number of random numbers to generate
 * @param max Upper bound (exclusive)
 * @return Container with generated random numbers
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, uint32>::type
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max);

/**
 * @brief [WRAPPER] Generate random discrete numbers in [0, max)
 * @param key Key
 * @param count Number of random numbers to generate
 * @param max Upper bound (exclusive)
 * @param numbers Output container for random numbers
 * @return void
 */
template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
_GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max, typename DeviceContainer<Policy, uint32>::ref_type numbers);


/**
 * @brief [DELEGATE] Transform uint32 random numbers to discrete [0, max)
 * @param u32_numbers Input uint32 random numbers
 * @param max Upper bound (exclusive)
 * @param numbers Output discrete random numbers
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_TransformDiscrete(typename DeviceContainer<Policy, uint32>::ref_const_type u32_numbers, const uint32 max, typename DeviceContainer<Policy, uint32>::ref_type numbers);
