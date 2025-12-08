#include <iostream>
#include <span>
#include <vector>
#include <execution>
#include <immintrin.h>
#include <ranges>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "KQS.Simulator.hpp"
#include "KQS.Complex.hpp"
#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
inline
void
Run_(std::span<uint> StateCounts, std::span<const LComplex> StateAmplitudes, const uint NumShots) {
    auto start = clock_type::now();
    const auto [res, ims] = DeinterleaveAoSLComplex<Policy>(StateAmplitudes);
    auto end = clock_type::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "DeinterleaveAoSLComplex took " << dur << " ms" << std::endl;
    
    start = clock_type::now();
    const auto probs = CalculateProbabilities<Policy>(res, ims);
    end = clock_type::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CalculateProbabilities took " << dur << " ms" << std::endl;

    start = clock_type::now();
    const auto table = BuildAliasTable<Policy>(probs);
    end = clock_type::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "BuildAliasTable took " << dur << " ms" << std::endl;

    start = clock_type::now();
    auto samples = SampleAliasTable<Policy, Algorithm>(table, NumShots);
    end = clock_type::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "SampleAliasTable took " << dur << " ms" << std::endl;

    start = clock_type::now();
    FlushSamples<Policy>(StateCounts, samples);
    end = clock_type::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "FlushSamples took " << dur << " ms" << std::endl;
}


std::vector<LComplex> GenerateUniformStateAmplitudes(size_t qubits) {
    const size_t numStates = 1ul << qubits;
    std::vector<LComplex> stateAmplitudes(numStates);
    const double amplitude = 1.0 / std::sqrt(static_cast<double>(numStates));
    for (size_t i = 0; i < numStates; ++i) {
        stateAmplitudes[i] = { amplitude, 0.0 };
    }
    return stateAmplitudes;
}


#ifndef EXECUTION_POLICY
#define EXECUTION_POLICY Sequential
#endif

constexpr ExecutionPolicy Policy = ExecutionPolicy::EXECUTION_POLICY;
constexpr PrngAlgorithm Algorithm = PrngAlgorithm::Philox;

int main() {
    auto stateAmplitudes = GenerateUniformStateAmplitudes(26);
    std::vector<uint> stateCounts(stateAmplitudes.size(), 0);
    const uint numShots = 1024*1024*256;
    Run_<Policy, Algorithm>(stateCounts, stateAmplitudes, numShots);

    // AlignedVector64<uint32> test1(1024*1024*1024);
    // AlignedVector64<uint32> test2(1024*1024*1024);
    // AlignedVector64<uint32> test3(1024*1024*1024);

    // const auto start = clock_type::now();
    // // 32 elements = grain of parallelism
    // // 8 elements = 256-bit AVX2 register
    // const auto grain = 1;
    // const auto idxes = std::views::iota(size_t{0}, test1.size() / 8 / grain) | std::views::transform([] (size_t i) { return i * 8 * grain; });
    // //std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
    // std::for_each(std::execution::par, idxes.begin(), idxes.end(),
    //     [&] (size_t i) {
    //         for (size_t j = 0; j < grain; ++j) {
    //             const size_t offset = i + j * 8;
    //             const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&test1[offset]));
    //             const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&test2[offset]));
    //             const __m256i c = _mm256_add_epi32(a, b);
    //             //_mm256_store_si256(reinterpret_cast<__m256i*>(&test3[offset]), c);
    //             _mm256_stream_si256(reinterpret_cast<__m256i*>(&test3[offset]), c);
    //         }
    //     }
    // );
    // const auto end = clock_type::now();
    // auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "Simple addition took " << dur << " ms" << std::endl;

    // AlignedVector64<uint32_t> test1(1024ull * 1024ull * 1024ull);
    // AlignedVector64<uint32_t> test2(1024ull * 1024ull * 1024ull);
    // AlignedVector64<uint32_t> test3(1024ull * 1024ull * 1024ull);

    // const size_t N = test1.size();
    // const size_t vec_size = 8;   // 8 × uint32 per AVX2 register

    // const auto start = clock_type::now();

    // // TBB automatically chooses chunk size
    // tbb::parallel_for(
    //     tbb::blocked_range<size_t>(0, N),
    //     [&](const tbb::blocked_range<size_t>& r)
    //     {
    //         size_t begin = r.begin();
    //         size_t end   = r.end();

    //         // Process AVX2 chunks
    //         for (size_t i = begin; i + vec_size <= end; i += vec_size) {
    //             __m256i a = _mm256_load_si256((__m256i*)&test1[i]);
    //             __m256i b = _mm256_load_si256((__m256i*)&test2[i]);
    //             __m256i c = _mm256_add_epi32(a, b);
    //             _mm256_store_si256((__m256i*)&test3[i], c);
    //         }
    //     }
    // );

    // const auto end = clock_type::now();
    // auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // std::cout << "Simple addition took " << dur << " ms\n";

    return 0;
}
