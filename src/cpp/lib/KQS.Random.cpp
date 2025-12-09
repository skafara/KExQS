#include "KQS.Random.hpp"
#include "KQS.CLManager.hpp"

#include <numeric>
#include <execution>
#include <ranges>
#include <immintrin.h>
#include <bit>
#include <span>
#include <random>
#include <filesystem>
#include <fstream>
#include <iostream>


#ifndef RANDOMORG_FILES_PATH
#define RANDOMORG_FILES_PATH "data/randomorg"
#endif


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(std::span<const double> probs) {
    const size_t n = probs.size();

    AlignedVector64<double> scaled(n);
    BenchmarkedFuncRun("_Scale",
        [&] () {
            _Scale<Policy>(probs, scaled);
        }
    );

    AlignedVector64<size_t> small;
    small.reserve(n);
    AlignedVector64<size_t> large;
    large.reserve(n);

    // Partition into small / large
    for (size_t i = 0; i < n; ++i) {
        if (scaled[i] < 1.0) {
            small.push_back(i);
        }
        else {
            large.push_back(i);
        }
    }

    // Main algorithm
    AlignedVector64<double> probs_(n);
    AlignedVector64<uint32> aliases_(n);
    while (!small.empty() && !large.empty()) {
        const size_t s = small.back();
        small.pop_back();
        const size_t l = large.back();
        large.pop_back();
        
        probs_[s] = scaled[s];
        aliases_[s] = l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        
        if (scaled[l] < 1.0) {
            small.push_back(l);
        }
        else {
            large.push_back(l);
        }
    }

    // Whatever remains
    for (const size_t i : large) {
        probs_[i] = 1.0;
    }
    for (const size_t i : small) {
        probs_[i] = 1.0;
    }

    return {probs_, aliases_};
}


template <>
void
_Scale<ExecutionPolicy::Sequential>(std::span<const double> probs, std::span<double> scaled) {
    const size_t n = probs.size();
    const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    const auto idxes = std::views::iota(size_t{0}, n);
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            scaled[i] = probs[i] * n / sum;
        }
    );
}

template <>
void
_Scale<ExecutionPolicy::Parallel>(std::span<const double> probs, std::span<double> scaled) {
    const size_t n = probs.size();
    const double sum = std::reduce(std::execution::par_unseq, probs.begin(), probs.end(), 0.0);
    const auto idxes = std::views::iota(size_t{0}, n);
    std::for_each(std::execution::par_unseq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            scaled[i] = probs[i] * n / sum; // TODO vectorize?
        }
    );
}

template <>
void
_Scale<ExecutionPolicy::Accelerated>(std::span<const double> probs, std::span<double> scaled) {
    _Scale<ExecutionPolicy::Parallel>(probs, scaled);
}


template
AliasTable
BuildAliasTable<ExecutionPolicy::Sequential>(std::span<const double> probs);

template
AliasTable
BuildAliasTable<ExecutionPolicy::Parallel>(std::span<const double> probs);

template
AliasTable
BuildAliasTable<ExecutionPolicy::Accelerated>(std::span<const double> probs);


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
AlignedVector64<uint32>
SampleAliasTable(const AliasTable &table, const uint NumShots) {
    AlignedVector64<uint32> samples(NumShots);

    // TODO seed management
    const auto r_bins = GenerateRandomDiscrete<Policy, Algorithm>(42ul, NumShots, table.Probs.size());
    const auto r_rands = GenerateRandomContinuous<Policy, Algorithm>(43ul, NumShots);

    if constexpr (Policy == ExecutionPolicy::Accelerated) {
        _SampleAliasTable<Policy>(table, r_bins, r_rands, samples);
    }
    else {
        BenchmarkedFuncRun("_SampleAliasTable",
            [&] () {
                _SampleAliasTable<Policy>(table, r_bins, r_rands, samples);
            }
        );
    }
    return samples;
}


template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Sequential, PrngAlgorithm::Philox>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Parallel, PrngAlgorithm::Philox>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Accelerated, PrngAlgorithm::Philox>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Sequential, PrngAlgorithm::MT19937>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Parallel, PrngAlgorithm::MT19937>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Accelerated, PrngAlgorithm::MT19937>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Sequential, PrngAlgorithm::RandomOrg>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Parallel, PrngAlgorithm::RandomOrg>(const AliasTable &table, const uint NumShots);

template
AlignedVector64<uint32>
SampleAliasTable<ExecutionPolicy::Accelerated, PrngAlgorithm::RandomOrg>(const AliasTable &table, const uint NumShots);


template <>
inline
void
_SampleAliasTable<ExecutionPolicy::Sequential>(const AliasTable &table, typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::const_ref_type bins, typename DeviceContainer<ExecutionPolicy::Sequential, double>::const_ref_type rands, std::span<uint32> samples) {
    const auto idxes = std::views::iota(size_t{0}, bins.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            const uint32 bin = bins[i];
            const double rand = rands[i];
            samples[i] = (rand < table.Probs[bin]) ? bin : table.Aliases[bin];
        }
    );
}

template <>
inline
void
_SampleAliasTable<ExecutionPolicy::Parallel>(const AliasTable &table, typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::const_ref_type bins, typename DeviceContainer<ExecutionPolicy::Parallel, double>::const_ref_type rands, std::span<uint32> samples) {
    const auto idxes = std::views::iota(size_t{0}, bins.size());
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            const uint32 bin = bins[i];
            const double rand = rands[i];
            samples[i] = (rand < table.Probs[bin]) ? bin : table.Aliases[bin];
        }
    );
}

template <>
inline
void
_SampleAliasTable<ExecutionPolicy::Accelerated>(const AliasTable &table, typename DeviceContainer<ExecutionPolicy::Accelerated, uint32>::const_ref_type bins, typename DeviceContainer<ExecutionPolicy::Accelerated, double>::const_ref_type rands, std::span<uint32> samples) {
    // TODO heavy cleanup
    // TODO memory
    CLManager &clManager = CLManager::Instance();
    cl::Kernel &kernel = clManager.GetKernel("_SampleAliasTable");
    
    cl::Buffer probsBuffer(clManager.GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, table.Probs.size() * sizeof(double), const_cast<double*>(table.Probs.data()));
    cl::Buffer aliasesBuffer(clManager.GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, table.Aliases.size() * sizeof(uint32), const_cast<uint32*>(table.Aliases.data()));
    cl::Buffer samplesBuffer(clManager.GetContext(), CL_MEM_WRITE_ONLY, samples.size() * sizeof(uint32));

    kernel.setArg(0, probsBuffer);
    kernel.setArg(1, aliasesBuffer);
    kernel.setArg(2, bins);
    kernel.setArg(3, rands);
    kernel.setArg(4, samplesBuffer);

    BenchmarkedKernelRun("_SampleAliasTable",
        [&] () {
            cl::Event event;
            clManager.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(samples.size()), cl::NullRange, nullptr, &event);
            return event;
        }
    );
    clManager.GetCommandQueue().enqueueReadBuffer(samplesBuffer, CL_TRUE, 0, samples.size() * sizeof(uint32), samples.data());
}


template <std::random_access_iterator Iterator>
inline
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, Iterator out) {
    constexpr uint64 M0 = 0xD2511F53u;
    constexpr uint64 M1 = 0xCD9E8D57u;
    constexpr uint32 W0 = 0x9E3779B9u;
    constexpr uint32 W1 = 0xBB67AE85u;

    uint32 k0 = static_cast<uint32>(key);
    uint32 k1 = static_cast<uint32>(key >> 32);

    uint32 x0 = static_cast<uint32>(counter);
    uint32 x1 = static_cast<uint32>(counter >> 32);
    uint32 x2 = 0u;
    uint32 x3 = 0u;

    for (int round = 0; round < 10; ++round) {
        const uint64 p0 = M0 * x0;
        const uint64 p1 = M1 * x2;

        const uint32 hi0 = static_cast<uint32>(p0 >> 32);
        const uint32 hi1 = static_cast<uint32>(p1 >> 32);
        const uint32 lo0 = static_cast<uint32>(p0);
        const uint32 lo1 = static_cast<uint32>(p1);

        const uint32 y0 = hi1 ^ x1 ^ k0;
        const uint32 y1 = lo1;
        const uint32 y2 = hi0 ^ x3 ^ k1;
        const uint32 y3 = lo0;

        x0 = y0;
        x1 = y1;
        x2 = y2;
        x3 = y3;

        k0 += W0;
        k1 += W1;
    }

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = x3;
}


inline __m256i mulhi_epu32_avx2(const __m256i a, const __m256i b) noexcept {
    const __m256i prod_even = _mm256_mul_epu32(a, b);
    const __m256i prod_odd  = _mm256_mul_epu32(_mm256_srli_si256(a, 4), _mm256_srli_si256(b, 4));
    const __m256i hi_even = _mm256_srli_epi64(prod_even, 32);
    const __m256i hi_odd  = _mm256_srli_epi64(prod_odd,  32);
    return _mm256_blend_epi32(hi_even, _mm256_slli_si256(hi_odd, 4), 0b10101010);
}


template <std::random_access_iterator Iterator, std::ranges::input_range Range>
requires std::same_as<std::ranges::range_value_t<Range>, uint64>
inline
void
GeneratePhilox8x4x32_10(const uint64 key, Range counters, Iterator out) {
    constexpr uint32 M0 = 0xD2511F53u;
    constexpr uint32 M1 = 0xCD9E8D57u;
    constexpr uint32 W0 = 0x9E3779B9u;
    constexpr uint32 W1 = 0xBB67AE85u;

    __m256i K0 = _mm256_set1_epi32(static_cast<uint32>(key));
    __m256i K1 = _mm256_set1_epi32(static_cast<uint32>(key >> 32));
    const __m256i W0v = _mm256_set1_epi32(W0);
    const __m256i W1v = _mm256_set1_epi32(W1);
    const __m256i M0v = _mm256_set1_epi32(M0);
    const __m256i M1v = _mm256_set1_epi32(M1);

    alignas(64) std::array<uint32, 8> ctr_lo, ctr_hi;
    for (size_t i = 0; i < 8; ++i) {
        const uint64 ctr = counters[i];
        ctr_lo[i] = static_cast<uint32>(ctr);
        ctr_hi[i] = static_cast<uint32>(ctr >> 32);
    }

    __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ctr_lo.data()));
    __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ctr_hi.data()));
    __m256i x2 = _mm256_setzero_si256();
    __m256i x3 = _mm256_setzero_si256();

    for (int round = 0; round < 10; ++round) {
        const __m256i p0_lo = _mm256_mullo_epi32(M0v, x0);
        const __m256i p0_hi = mulhi_epu32_avx2(M0v, x0);
        const __m256i p1_lo = _mm256_mullo_epi32(M1v, x2);
        const __m256i p1_hi = mulhi_epu32_avx2(M1v, x2);

        const __m256i hi0 = p0_hi;
        const __m256i hi1 = p1_hi;
        const __m256i lo0 = p0_lo;
        const __m256i lo1 = p1_lo;

        const __m256i y0 = _mm256_xor_si256(_mm256_xor_si256(hi1, x1), K0);
        const __m256i y1 = lo1;
        const __m256i y2 = _mm256_xor_si256(_mm256_xor_si256(hi0, x3), K1);
        const __m256i y3 = lo0;

        x0 = y0;
        x1 = y1;
        x2 = y2;
        x3 = y3;

        K0 = _mm256_add_epi32(K0, W0v);
        K1 = _mm256_add_epi32(K1, W1v);
    }

    // SoA -> AoS
    // Now we have:
    // x0 = [b0.x0, b1.x0, b2.x0, b3.x0 | b4.x0, b5.x0, b6.x0, b7.x0]
    // x1 = [b0.x1, b1.x1, b2.x1, b3.x1 | b4.x1, b5.x1, b6.x1, b7.x1]
    // x2 = [b0.x2, b1.x2, b2.x2, b3.x2 | b4.x2, b5.x2, b6.x2, b7.x2]
    // x3 = [b0.x3, b1.x3, b2.x3, b3.x3 | b4.x3, b5.x3, b6.x3, b7.x3]
    // We want:
    // out = [b0.x0, b0.x1, b0.x2, b0.x3 | b1.x0, b1.x1, b1.x2, b1.x3 | ... | b7.x0, b7.x1, b7.x2, b7.x3]

    const __m256i t0 = _mm256_unpacklo_epi32(x0, x1); // [b0.x0,b0.x1, b1.x0,b1.x1 | b4.x0,b4.x1, b5.x0,b5.x1]
    const __m256i t1 = _mm256_unpackhi_epi32(x0, x1); // [b2.x0,b2.x1, b3.x0,b3.x1 | b6.x0,b6.x1, b7.x0,b7.x1]
    const __m256i t2 = _mm256_unpacklo_epi32(x2, x3); // [b0.x2,b0.x3, b1.x2,b1.x3 | b4.x2,b4.x3, b5.x2,b5.x3]
    const __m256i t3 = _mm256_unpackhi_epi32(x2, x3); // [b2.x2,b2.x3, b3.x2,b3.x3 | b6.x2,b6.x3, b7.x2,b7.x3]

    // Build pairs (0,1), (2,3), (4,5), (6,7)
    const __m256i p0 = _mm256_unpacklo_epi64(t0, t2); // [b0.x[0..3] | b4.x[0..3]]
    const __m256i p1 = _mm256_unpacklo_epi64(t1, t3); // [b2.x[0..3] | b6.x[0..3]]
    const __m256i p2 = _mm256_unpackhi_epi64(t0, t2); // [b1.x[0..3] | b5.x[0..3]]
    const __m256i p3 = _mm256_unpackhi_epi64(t1, t3); // [b3.x[0..3] | b7.x[0..3]]

    const __m256i o0 = _mm256_permute2x128_si256(p0, p2, 0x20); // [b0 | b1]
    const __m256i o1 = _mm256_permute2x128_si256(p1, p3, 0x20); // [b2 | b3]
    const __m256i o2 = _mm256_permute2x128_si256(p0, p2, 0x31); // [b4 | b5]
    const __m256i o3 = _mm256_permute2x128_si256(p1, p3, 0x31); // [b6 | b7]
    
    _mm256_store_si256(reinterpret_cast<__m256i*>(&out[0]), o0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(&out[8]), o1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(&out[16]), o2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(&out[24]), o3);
}


template <ExecutionPolicy Policy>
void
GenerateRandomUint32(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint32>::ref_type numbers) {
    _GenerateRandomUint32<Policy>(key, count, numbers);
}


template <>
inline
void
_GenerateRandomUint32<ExecutionPolicy::Sequential>(const uint64 key, const size_t count, typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::ref_type numbers) {
    const auto idxes = std::views::iota(uint64{0}, uint64{count / 4});
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 4;
            GeneratePhilox4x32_10(key, i, &numbers[offset]);
        }
    );

    const size_t rem = count % 4;
    if (rem > 0) {  
        const auto offset = count - rem;
        std::array<uint32, 4> numbers_;
        GeneratePhilox4x32_10(key, count / 4, numbers_.data());

        const auto idxes = std::views::iota(size_t{0}, rem);
        std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
            [&] (size_t i) {
                numbers[offset + i] = numbers_[i];
            }
        );
    }
}

template <>
inline
void
_GenerateRandomUint32<ExecutionPolicy::Parallel>(const uint64 key, const size_t count, typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::ref_type numbers) {
    const auto idxes = std::views::iota(uint64{0}, uint64{count / 32});
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 32;
            const auto counters = std::views::iota(i * 8, i * 8 + 8);
            GeneratePhilox8x4x32_10(key, counters, &numbers[offset]);
        }
    );

    const size_t rem = count % 32;
    if (rem > 0) {
        const auto offset = count - rem;
        const auto counters = std::views::iota(uint64{(count / 32) * 8}, uint64{(count / 32) * 8 + 8});
        GeneratePhilox8x4x32_10(key, counters, &numbers[offset]);
    }
}


template <ExecutionPolicy Policy>
void
GenerateRandomUint64(const uint64 key, const size_t count, typename DeviceContainer<Policy, uint64>::ref_type numbers) {
    _GenerateRandomUint64<Policy>(key, count, numbers);
}


template <>
inline
void
_GenerateRandomUint64<ExecutionPolicy::Sequential>(const uint64 key, const size_t count, typename DeviceContainer<ExecutionPolicy::Sequential, uint64>::ref_type numbers) {
    const auto Transform = [] (const uint32 hi, const uint32 lo) -> uint64 {
        return (static_cast<uint64>(hi) << 32) | static_cast<uint64>(lo);
    };

    const auto idxes = std::views::iota(uint64{0}, uint64{count / 2});
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 2;
            std::array<uint32, 4> numbers_;
            GeneratePhilox4x32_10(key, i, numbers_.data());
            numbers[offset + 0] = Transform(numbers_[0], numbers_[1]);
            numbers[offset + 1] = Transform(numbers_[2], numbers_[3]);
        }
    );

    const size_t rem = count % 2;
    if (rem > 0) {  
        const auto offset = count - rem;
        std::array<uint32, 4> numbers_;
        GeneratePhilox4x32_10(key, count / 2, numbers_.data());
        numbers[offset + 0] = Transform(numbers_[0], numbers_[1]);
    }
}

template <>
inline
void
_GenerateRandomUint64<ExecutionPolicy::Parallel>(const uint64 key, const size_t count, typename DeviceContainer<ExecutionPolicy::Parallel, uint64>::ref_type numbers) {
    const auto idxes = std::views::iota(uint64{0}, uint64{count / 16});
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 16;
            const auto counters = std::views::iota(i * 8, i * 8 + 8);
            std::array<uint32, 32> numbers_;
            GeneratePhilox8x4x32_10(key, counters, numbers_.data());
            for (size_t j = 0; j < 16; ++j) {
                numbers[offset + j] = (static_cast<uint64>(numbers_[j * 2]) << 32) | static_cast<uint64>(numbers_[j * 2 + 1]);
            }
        }
    );

    const size_t rem = count % 16;
    if (rem > 0) {
        const auto offset = count - rem;
        const auto counters = std::views::iota(uint64{(count / 16) * 8}, uint64{(count / 16) * 8 + 8});
        std::array<uint32, 32> numbers_;
        GeneratePhilox8x4x32_10(key, counters, numbers_.data());
        for (size_t i = 0; i < rem; ++i) {
            numbers[offset + i] = (static_cast<uint64>(numbers_[i * 2]) << 32) | static_cast<uint64>(numbers_[i * 2 + 1]);
        }
    }
}


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, double>::type
GenerateRandomContinuous(const uint64 key, const size_t count) {
    if constexpr (Algorithm == PrngAlgorithm::Philox) {
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            // TODO heavy cleanup
            // TODO memory
            CLManager &clManager = CLManager::Instance();
            cl::Kernel &kernel = clManager.GetKernel("GenerateRandomContinuous");
            
            cl::Buffer outBuffer(clManager.GetContext(), CL_MEM_WRITE_ONLY, count * sizeof(double));

            kernel.setArg(0, key);
            kernel.setArg(1, outBuffer);

            BenchmarkedKernelRun("GenerateRandomContinuous",
                [&] () {
                    cl::Event event;
                    clManager.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(count / 2), cl::NullRange, nullptr, &event);
                    return event;
                }
            );

            // TODO remainding elements
            return outBuffer;
        }
        else {
            typename DeviceContainer<Policy, uint64>::type u64_numbers(count);
            typename DeviceContainer<Policy, double>::type numbers(count);

            BenchmarkedFuncRun("GenerateRandomContinuous",
                [&] () {
                    GenerateRandomUint64<Policy>(key, count, u64_numbers);
                    _GenerateRandomContinuous<Policy>(u64_numbers, numbers);
                }
            );
            return numbers;
        }
    }
    else if constexpr (Algorithm == PrngAlgorithm::MT19937) {
        std::mt19937_64 mt{key};
        typename DeviceContainer<ExecutionPolicy::Sequential, uint64>::type u64_numbers(count);
        for (size_t i = 0; i < count; ++i) {
            u64_numbers[i] = mt();
        }
        
        typename DeviceContainer<ExecutionPolicy::Sequential, double>::type numbers(count);
        _GenerateRandomContinuous<ExecutionPolicy::Sequential>(u64_numbers, numbers);
        
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            cl::Buffer outBuffer(CLManager::Instance().GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(double), numbers.data());
            return outBuffer;
        } else {
            return numbers;
        }
    }
    else if constexpr (Algorithm == PrngAlgorithm::RandomOrg) {
        std::filesystem::directory_iterator it_paths(RANDOMORG_FILES_PATH);
        typename DeviceContainer<ExecutionPolicy::Sequential, uint64>::type u64_numbers(count);

        const auto LoadNumbers = [] (std::string path) {
            const auto fileSize = std::filesystem::file_size(path);
            std::vector<uint64> data(fileSize / sizeof(uint64));
            std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(data.data()), fileSize);
            return data;
        };

        std::vector<uint64> loaded;
        auto it_loaded = loaded.begin();
        for (size_t i = 0; i < count; ++i) {
            if (it_loaded == loaded.end()) {
                ++it_paths;
                loaded = LoadNumbers(it_paths->path().string());
                it_loaded = loaded.begin();
            }

            u64_numbers[i] = *it_loaded;
            ++it_loaded;
        }

        typename DeviceContainer<ExecutionPolicy::Sequential, double>::type numbers(count);
        _GenerateRandomContinuous<ExecutionPolicy::Sequential>(u64_numbers, numbers);
        
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            cl::Buffer outBuffer(CLManager::Instance().GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(double), numbers.data());
            return outBuffer;
        } else {
            return numbers;
        }
    }
    else {
        throw std::runtime_error("Unknown PRNG Algorithm");
    }
}


template <>
inline
void
_GenerateRandomContinuous<ExecutionPolicy::Sequential>(typename DeviceContainer<ExecutionPolicy::Sequential, uint64>::const_ref_type u64_numbers, typename DeviceContainer<ExecutionPolicy::Sequential, double>::ref_type numbers) {
    constexpr double INV2P53 = 1.0 / static_cast<double>(1ull << 53);

    const auto idxes = std::views::iota(size_t{0}, numbers.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&](size_t i) {
            const uint64 mantissa = u64_numbers[i] >> 11;
            numbers[i] = static_cast<double>(mantissa) * INV2P53;
        }
    );
}

template <>
inline
void
_GenerateRandomContinuous<ExecutionPolicy::Parallel>(typename DeviceContainer<ExecutionPolicy::Parallel, uint64>::const_ref_type u64_numbers, typename DeviceContainer<ExecutionPolicy::Parallel, double>::ref_type numbers) {
    constexpr double INV2P53 = 1.0 / static_cast<double>(1ull << 53);

    const auto idxes = std::views::iota(size_t{0}, numbers.size());
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&](size_t i) {
            // TODO vectorize?
            const uint64 mantissa = u64_numbers[i] >> 11;
            numbers[i] = static_cast<double>(mantissa) * INV2P53;
        }
    );
}


template <ExecutionPolicy Policy, PrngAlgorithm Algorithm>
DeviceContainer<Policy, uint32>::type
GenerateRandomDiscrete(const uint64 key, const size_t count, const uint32 max) {
    if constexpr (Algorithm == PrngAlgorithm::Philox) {
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            // TODO heavy cleanup
            // TODO memory
            CLManager &clManager = CLManager::Instance();
            cl::Kernel &kernel = clManager.GetKernel("GenerateRandomDiscrete");
            
            cl::Buffer outBuffer(clManager.GetContext(), CL_MEM_WRITE_ONLY, count * sizeof(uint32));

            kernel.setArg(0, key);
            kernel.setArg(1, max);
            kernel.setArg(2, outBuffer);

            BenchmarkedKernelRun("GenerateRandomDiscrete",
                [&] () {
                    cl::Event event;
                    clManager.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(count / 4), cl::NullRange, nullptr, &event);
                    return event;
                }
            );

            // TODO remainding elements
            return outBuffer;
        } else {
            typename DeviceContainer<Policy, uint32>::type u32_numbers(count);
            typename DeviceContainer<Policy, uint32>::type numbers(count);

            BenchmarkedFuncRun("GenerateRandomDiscrete",
                [&] () {
                    GenerateRandomUint32<Policy>(key, count, u32_numbers);
                    _GenerateRandomDiscrete<Policy>(u32_numbers, max, numbers);
                }
            );
            return numbers;
        }
    } else if constexpr (Algorithm == PrngAlgorithm::MT19937) {
        std::mt19937 mt{static_cast<uint32>(key)};
        typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::type u32_numbers(count);
        for (size_t i = 0; i < count; ++i) {
            u32_numbers[i] = mt();
        }
        
        typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::type numbers(count);
        _GenerateRandomDiscrete<ExecutionPolicy::Sequential>(u32_numbers, max, numbers);
        
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            cl::Buffer outBuffer(CLManager::Instance().GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32), numbers.data());
            return outBuffer;
        } else {
            return numbers;
        }
    }
    else if constexpr (Algorithm == PrngAlgorithm::RandomOrg) {
        std::filesystem::directory_iterator it_paths(RANDOMORG_FILES_PATH);
        typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::type u32_numbers(count);

        const auto LoadNumbers = [] (std::string path) {
            const auto fileSize = std::filesystem::file_size(path);
            std::vector<uint32> data(fileSize / sizeof(uint32));
            std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(data.data()), fileSize);
            return data;
        };

        std::vector<uint32> pool;
        for (; it_paths != std::filesystem::end(it_paths); ++it_paths) {
            const auto loaded = LoadNumbers(it_paths->path().string());
            pool.insert(pool.end(), loaded.begin(), loaded.end());
        }

        typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::type numbers(count);
        std::sample(pool.begin(), pool.end(), u32_numbers.begin(), count, std::mt19937{static_cast<uint32>(key)});
        _GenerateRandomDiscrete<ExecutionPolicy::Parallel>(u32_numbers, max, numbers);
        
        if constexpr (Policy == ExecutionPolicy::Accelerated) {
            cl::Buffer outBuffer(CLManager::Instance().GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32), numbers.data());
            return outBuffer;
        } else {
            return numbers;
        }
    }
    else {
        throw std::runtime_error("Unknown PRNG Algorithm");
    }    
}


template <>
inline
void
_GenerateRandomDiscrete<ExecutionPolicy::Sequential>(typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::const_ref_type u32_numbers, const uint32 max, typename DeviceContainer<ExecutionPolicy::Sequential, uint32>::ref_type numbers) {
    const auto idxes = std::views::iota(size_t{0}, numbers.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&](size_t i) {
            numbers[i] = static_cast<uint32>(static_cast<uint64>(u32_numbers[i]) * static_cast<uint64>(max) >> 32);
        }
    );
}

template <>
inline
void
_GenerateRandomDiscrete<ExecutionPolicy::Parallel>(typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::const_ref_type u32_numbers, const uint32 max, typename DeviceContainer<ExecutionPolicy::Parallel, uint32>::ref_type numbers) {
    const auto idxes = std::views::iota(size_t{0}, numbers.size());
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&](size_t i) { // TODO vectorize?
            numbers[i] = static_cast<uint32>(static_cast<uint64>(u32_numbers[i]) * static_cast<uint64>(max) >> 32);
        }
    );
}
