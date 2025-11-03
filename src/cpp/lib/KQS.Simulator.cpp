#include <span>
#include <vector>
#include <complex>
#include <bit>
#include <random>
#include <algorithm>
#include <execution>
#include <ranges>
#include <immintrin.h>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"


constexpr ExecutionPolicy Policy = ExecutionPolicy::Sequential;


inline
double
CalculateProbability(const double re, const double im) {
    return re * re + im * im;
}


inline
__m256d
CalculateProbability(const __m256d re, const __m256d im) {
    const __m256d re2 = _mm256_mul_pd(re, re); // [Re0^2, Re1^2, Re2^2, Re3^2]
    const __m256d im2 = _mm256_mul_pd(im, im); // [Im0^2, Im1^2, Im2^2, Im3^2]
    const __m256d prob = _mm256_add_pd(re2, im2); // [Re0^2 + Im0^2, ..., Re3^2 + Im3^2]
    return prob;
}


template <ExecutionPolicy Policy>
std::vector<double>
CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims) {
    std::vector<double> probs(res.size()); // TODO align to 32 bytes
    _CalculateProbabilities<Policy>(res, ims, probs);
    return probs;
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Sequential>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    const auto idxes = std::views::iota(size_t{0}, res.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            probs[i] = CalculateProbability(res[i], ims[i]);
        }
    );
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Parallel>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    const auto idxes = std::views::iota(size_t{0}, res.size() / 4) | std::views::transform([] (size_t i) { return i * 4; });
    // 1 Block = 4 Complex = 8 doubles = 2x 256-bit AVX2 registers
    // [Re0, Re1, Re2, Re3]
    // [Im0, Im1, Im2, Im3]
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            const __m256d re = _mm256_load_pd(&res[i]); // [Re0, Re1, Re2, Re3]
            const __m256d im = _mm256_load_pd(&ims[i]); // [Im0, Im1, Im2, Im3]
            const __m256d prob = CalculateProbability(re, im); // [Re0^2 + Im0^2, ..., Re3^2 + Im3^2]
            _mm256_store_pd(&probs[i], prob);
        }
    );

    const size_t rem = res.size() % 4;
    if (rem > 0) {
        const auto offset = res.size() - rem;
        const auto idxes = std::views::iota(size_t{offset}, res.size());
        std::for_each(std::execution::par, idxes.begin(), idxes.end(),
            [&] (size_t i) {
                probs[i] = CalculateProbability(res[i], ims[i]);
            }
        );
    }
}


template <>
void
_CalculateProbabilities<ExecutionPolicy::Accelerated>(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs) {
    throw std::runtime_error("Not implemented"); // TODO implement GPU accelerated version
}


template <>
void
FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> StateCounts, const std::vector<uint> &samples) {
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint sample) {
            StateCounts[sample]++;
        }
    );
}


template <>
void
FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> StateCounts, const std::vector<uint32_t> &samples) {
    std::for_each(std::execution::par, samples.begin(), samples.end(),
        [&](uint32_t sample) {
            // atomic increment to avoid race conditions
            std::atomic_ref<uint>(StateCounts[sample]).fetch_add(1, std::memory_order_relaxed);
        }
    ); // TODO without atomic operations using thread-local StateCounts and then reduce
}


template <>
void
FlushSamples<ExecutionPolicy::Accelerated>(std::span<uint> StateCounts, const std::vector<uint> &samples) {
    FlushSamples<ExecutionPolicy::Parallel>(StateCounts, samples);
}


template <ExecutionPolicy Policy>
void
Run(const std::span<uint> StateCounts, const std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    const auto table = BuildAliasTable<Policy>(probs);
    const auto samples = SampleAliasTable<Policy>(table, NumShots);
    FlushSamples<Policy>(StateCounts, samples);
}


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
) {
    const std::span<const LComplex> StateAmplitudes(AStateAmplitudes, ANumStates);
    std::span<uint> StateCounts(AStateCounts, ANumStates);
    Run<Policy>(StateCounts, StateAmplitudes, ANumShots);
}
