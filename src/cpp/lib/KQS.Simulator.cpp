#include <span>
#include <vector>
#include <complex>
#include <bit>
#include <random>
#include <algorithm>
#include <execution>
#include <ranges>
#include <immintrin.h>
#include <iostream>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"


//static std::mt19937 gen(std::random_device{}());

constexpr ExecutionPolicy Policy = ExecutionPolicy::Parallel;


template <ExecutionPolicy Policy>
std::vector<double>
CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims) {
    std::vector<double> probs(res.size()); // TODO align to 32 bytes
    
    if (res.size() == 2) {
        _CalculateProbabilities<ExecutionPolicy::Sequential>(res, ims, probs);
    } else {
        _CalculateProbabilities<Policy>(res, ims, probs);
    }

    return probs;
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Sequential>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    const auto idxes = std::views::iota(size_t{0}, res.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            probs[i] = res[i] * res[i] + ims[i] * ims[i];
        }
    );
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Parallel>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    const auto block_idxes = std::views::iota(size_t{0}, res.size() / 4);
    // 1 Block = 4 Complex = 8 doubles = 2x 256-bit AVX2 registers
    // [Re0, Re1, Re2, Re3]
    // [Im0, Im1, Im2, Im3]
    std::for_each(std::execution::par, block_idxes.begin(), block_idxes.end(),
        [&] (size_t block_idx) {
            const auto i = block_idx * 4;
            __m256d re_ = _mm256_load_pd(&res[i]); // [Re0, Re1, Re2, Re3]
            __m256d im_ = _mm256_load_pd(&ims[i]); // [Im0, Im1, Im2, Im3]
            re_ = _mm256_mul_pd(re_, re_); // [Re0^2, Re1^2, Re2^2, Re3^2]
            im_ = _mm256_mul_pd(im_, im_); // [Im0^2, Im1^2, Im2^2, Im3^2]
            const __m256d p_ = _mm256_add_pd(re_, im_); // [Re0^2 + Im0^2, ..., Re3^2 + Im3^2]
            _mm256_store_pd(&probs[i], p_);
        }
    );
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Accelerated>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    throw std::runtime_error("Not implemented"); // TODO implement GPU accelerated version
}


template <>
void
FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> counts, const std::vector<uint32_t> &samples) {
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint32_t sample) {
            counts[sample]++;
        }
    );
}


template <>
void
FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> counts, const std::vector<uint32_t> &samples) {
    std::for_each(std::execution::par, samples.begin(), samples.end(),
        [&](uint32_t sample) {
            // atomic increment to avoid race conditions
            std::atomic_ref<uint>(counts[sample]).fetch_add(1, std::memory_order_relaxed);
        }
    ); // TODO without atomic operations using thread-local counts and then reduce
}


template <ExecutionPolicy Policy>
void
Run(const std::span<uint> AStateCounts, const std::span<const LComplex> AStateAmplitudes, const uint ANumShots) {
    auto [res, ims] = DeinterleaveAoSLComplex<Policy>(AStateAmplitudes);
    auto probs = CalculateProbabilities<Policy>(res, ims);

    const auto r_bins = GenerateRandomDiscrete<Policy>(1ull, ANumShots, AStateAmplitudes.size());
    const auto r_rands = GenerateRandomContinuous<Policy>(1ull, ANumShots);

    const auto table = BuildAliasTable<Policy>(probs);
    auto samples = SampleAliasTable<Policy>(table, r_bins, r_rands);
    FlushSamples<Policy>(AStateCounts, samples);
}


template <>
void
Run<ExecutionPolicy::Accelerated>(const std::span<uint> AStateCounts, const std::span<const LComplex> AStateAmplitudes, const uint ANumShots) {
    throw std::runtime_error("Not implemented"); // TODO implement GPU accelerated version
}


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    const std::span<const LComplex> amplitudes(AStateAmplitudes, ANumStates);
    std::span<uint> counts(AStateCounts, ANumStates);

    Run<Policy>(counts, amplitudes, ANumShots);

    /*auto [res, ims] = DeinterleaveAoSLComplex<Policy>(amplitudes);
    auto probs = CalculateProbabilities<Policy>(res, ims);*/

    /*std::vector<size_t> r_bins(ANumShots);
    std::vector<double> r_rands(ANumShots);
    std::uniform_int_distribution<size_t> u_bins(0, ANumStates - 1);
    std::uniform_real_distribution<double> u_01(0.0, 1.0);
    for (size_t i = 0; i < ANumShots; i++) {
        r_bins[i] = u_bins(gen);
        r_rands[i] = u_01(gen);
    }*/

    /*const auto r_bins = GenerateRandomDiscrete<Policy>(1ull, ANumShots, ANumStates);
    const auto r_rands = GenerateRandomContinuous<Policy>(1ull, ANumShots);

    const auto table = BuildAliasTable<Policy>(probs);
    auto samples = SampleAliasTable<Policy>(table, r_bins, r_rands);
    FlushSamples<Policy>(counts, samples);*/
    
    /*std::discrete_distribution<uint> dist(probs.begin(), probs.end());
    for (size_t i = 0; i < ANumShots; i++) {
        const uint state = dist(gen);
        AStateCounts[state]++;
    }*/

    /*uint64_t key = 1ull;
    auto counters = std::views::iota(uint64_t{0}, uint64_t{8});
    for (const auto counter : counters) {
        auto random_numbers = GeneratePhilox4x32_10(key, counter);
        for (const auto rn : random_numbers) {
            std::cout << rn << " ";
        }
    }
    std::cout << std::endl;

    std::vector<uint64_t> counters_vec(counters.begin(), counters.end());
    auto random_numbers = GeneratePhilox8x4x32_10(key, counters_vec);
    for (const auto rn : random_numbers) {
        std::cout << rn << " ";
    }
    std::cout << std::endl;*/

    /*auto random_numbers_ = GenerateRandomUint32<Policy>(1ull, 10);
    for (const auto rn : random_numbers_) {
        std::cout << rn << " ";
    }
    std::cout << std::endl;*/

    /*auto random_numbers_ = GenerateRandomContinuous<Policy>(1ull, 10);
    for (const auto rn : random_numbers_) {
        std::cout << rn << " ";
    }
    std::cout << std::endl;

    auto random_numbers__ = GenerateRandomDiscrete<Policy>(1ull, 100, 5);
    for (const auto rn : random_numbers__) {
        std::cout << rn << " ";
    }
    std::cout << std::endl;*/
}
