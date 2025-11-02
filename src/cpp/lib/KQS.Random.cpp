#include "KQS.Random.hpp"

#include <numeric>
#include <execution>
#include <ranges>
#include <immintrin.h>
#include <bit>
#include <span>


template <ExecutionPolicy Policy>
AliasTable
BuildAliasTable(const std::vector<double>& probs) {
    const size_t n = probs.size();
    std::vector<double> scaled(n);
    std::vector<size_t> small, large;
    small.reserve(n);
    large.reserve(n);

    // Normalize probabilities so they sum to 1
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    for (size_t i = 0; i < n; ++i)
        scaled[i] = probs[i] * n / sum; // scale so mean = 1

    // Partition into small / large
    for (size_t i = 0; i < n; ++i)
        (scaled[i] < 1.0 ? small : large).push_back(i);

    std::vector<double> prob(n);
    std::vector<uint32> alias(n);

    // Main algorithm
    while (!small.empty() && !large.empty()) {
        size_t s = small.back(); small.pop_back();
        size_t l = large.back(); large.pop_back();
        prob[s] = scaled[s];
        alias[s] = l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        (scaled[l] < 1.0 ? small : large).push_back(l);
    }

    // Whatever remains
    for (size_t i : large) prob[i] = 1.0;
    for (size_t i : small) prob[i] = 1.0;

    return {prob, alias};
}


template
AliasTable
BuildAliasTable<ExecutionPolicy::Sequential>(const std::vector<double>& probs);

template
AliasTable
BuildAliasTable<ExecutionPolicy::Parallel>(const std::vector<double>& probs);

template
AliasTable
BuildAliasTable<ExecutionPolicy::Accelerated>(const std::vector<double>& probs);


template <ExecutionPolicy Policy>
std::vector<uint32>
SampleAliasTable(const AliasTable &table, const std::vector<uint32>& bins, const std::vector<double>& rands) {
    std::vector<uint32> samples(bins.size());

    const auto idxes = std::views::iota(size_t{0}, bins.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            const uint32 bin = bins[i];
            const double rand = rands[i];
            samples[i] = (rand < table.Probs[bin]) ? bin : table.Aliases[bin];
        }
    );

    return samples;
}


template
std::vector<uint32>
SampleAliasTable<ExecutionPolicy::Sequential>(const AliasTable &table, const std::vector<uint32>& bins, const std::vector<double>& rands);

template
std::vector<uint32>
SampleAliasTable<ExecutionPolicy::Parallel>(const AliasTable &table, const std::vector<uint32>& bins, const std::vector<double>& rands);

template
std::vector<uint32>
SampleAliasTable<ExecutionPolicy::Accelerated>(const AliasTable &table, const std::vector<uint32>& bins, const std::vector<double>& rands);


template <std::random_access_iterator Iterator>
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, Iterator out) {
    constexpr uint32 M0 = 0xD2511F53u;
    constexpr uint32 M1 = 0xCD9E8D57u;
    constexpr uint32 W0 = 0x9E3779B9u;
    constexpr uint32 W1 = 0xBB67AE85u;

    uint32 k0 = static_cast<uint32>(key);
    uint32 k1 = static_cast<uint32>(key >> 32);

    uint32 x0 = static_cast<uint32>(counter);
    uint32 x1 = static_cast<uint32>(counter >> 32);
    uint32 x2 = 0u;
    uint32 x3 = 0u;

    for (int round = 0; round < 10; ++round) {
        const uint64 p0 = static_cast<uint64>(M0) * x0;
        const uint64 p1 = static_cast<uint64>(M1) * x2;

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


// AVX2 helper: unsigned high 32x32->64 multiply per lane: (a*b)>>32
inline __m256i mulhi_epu32_avx2(__m256i a, __m256i b) noexcept {
    __m256i prod_even = _mm256_mul_epu32(a, b);                               // lanes 0,2,4,6
    __m256i prod_odd  = _mm256_mul_epu32(_mm256_srli_si256(a, 4),
                                         _mm256_srli_si256(b, 4));            // lanes 1,3,5,7
    __m256i hi_even = _mm256_srli_epi64(prod_even, 32);
    __m256i hi_odd  = _mm256_srli_epi64(prod_odd,  32);
    return _mm256_blend_epi32(hi_even, _mm256_slli_si256(hi_odd, 4), 0b10101010);
}


template <std::random_access_iterator Iterator>
void
GeneratePhilox8x4x32_10(const uint64 key, const std::array<uint64, 8> &counters, Iterator out) {
    constexpr uint32 M0 = 0xD2511F53u;
    constexpr uint32 M1 = 0xCD9E8D57u;
    constexpr uint32 W0 = 0x9E3779B9u;
    constexpr uint32 W1 = 0xBB67AE85u;

    __m256i K0 = _mm256_set1_epi32(static_cast<int>(static_cast<uint32>(key)));
    __m256i K1 = _mm256_set1_epi32(static_cast<int>(static_cast<uint32>(key >> 32)));
    const __m256i W0v = _mm256_set1_epi32(static_cast<int>(W0));
    const __m256i W1v = _mm256_set1_epi32(static_cast<int>(W1));
    const __m256i M0v = _mm256_set1_epi32(static_cast<int>(M0));
    const __m256i M1v = _mm256_set1_epi32(static_cast<int>(M1));

    alignas(32) std::array<uint32, 8> c0{}, c1{}, c2{}, c3{};
    for (size_t i = 0; i < 8; ++i) {
        const uint64 ctr = counters[i];
        c0[i] = static_cast<uint32>(ctr);
        c1[i] = static_cast<uint32>(ctr >> 32);
        c2[i] = 0u;
        c3[i] = 0u;
    }

    __m256i x0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(c0.data()));
    __m256i x1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(c1.data()));
    __m256i x2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(c2.data()));
    __m256i x3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(c3.data()));

    // 10 rounds
    for (int r = 0; r < 10; ++r) {
        const __m256i lo0 = _mm256_mullo_epi32(M0v, x0);
        const __m256i hi0 = mulhi_epu32_avx2(M0v, x0);
        const __m256i lo1 = _mm256_mullo_epi32(M1v, x2);
        const __m256i hi1 = mulhi_epu32_avx2(M1v, x2);

        const __m256i y0 = _mm256_xor_si256(_mm256_xor_si256(hi1, x1), K0);
        const __m256i y1 = lo1;
        const __m256i y2 = _mm256_xor_si256(_mm256_xor_si256(hi0, x3), K1);
        const __m256i y3 = lo0;

        x0 = y0; x1 = y1; x2 = y2; x3 = y3;
        K0 = _mm256_add_epi32(K0, W0v);
        K1 = _mm256_add_epi32(K1, W1v);
    }

    // === SoA -> AoS transpose (correct sequential order: blocks 0..7) ===
    __m256i t0 = _mm256_unpacklo_epi32(x0, x1); // [b0.x0,b0.x1, b1.x0,b1.x1 | b4.x0,b4.x1, b5.x0,b5.x1]
    __m256i t1 = _mm256_unpackhi_epi32(x0, x1); // [b2.x0,b2.x1, b3.x0,b3.x1 | b6.x0,b6.x1, b7.x0,b7.x1]
    __m256i t2 = _mm256_unpacklo_epi32(x2, x3); // [b0.x2,b0.x3, b1.x2,b1.x3 | b4.x2,b4.x3, b5.x2,b5.x3]
    __m256i t3 = _mm256_unpackhi_epi32(x2, x3); // [b2.x2,b2.x3, b3.x2,b3.x3 | b6.x2,b6.x3, b7.x2,b7.x3]

    // Build pairs (0,1), (2,3), (4,5), (6,7)
    __m256i v01_lo = _mm256_unpacklo_epi64(t0, t2); // lower128: b0[0..3], upper128: b4[0..3]
    __m256i v23_lo = _mm256_unpacklo_epi64(t1, t3); // lower128: b2[0..3], upper128: b6[0..3]
    __m256i v01_hi = _mm256_unpackhi_epi64(t0, t2); // lower128: b1[0..3], upper128: b5[0..3]
    __m256i v23_hi = _mm256_unpackhi_epi64(t1, t3); // lower128: b3[0..3], upper128: b7[0..3]

    __m256i u0 = _mm256_permute2x128_si256(v01_lo, v01_hi, 0x20); // b0 | b1
    __m256i u1 = _mm256_permute2x128_si256(v23_lo, v23_hi, 0x20); // b2 | b3
    __m256i u2 = _mm256_permute2x128_si256(v01_lo, v01_hi, 0x31); // b4 | b5
    __m256i u3 = _mm256_permute2x128_si256(v23_lo, v23_hi, 0x31); // b6 | b7
    
    std::array<uint32, 32> out_;
    _mm256_store_si256(reinterpret_cast<__m256i*>(out_.data() +  0), u0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out_.data() +  8), u1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out_.data() + 16), u2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out_.data() + 24), u3);

    const auto idxes = std::views::iota(size_t{0}, size_t{32});
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            out[i] = out_[i];
        }
    );
}


template <>
std::vector<uint32>
GenerateRandomUint32<ExecutionPolicy::Sequential>(const uint64 key, const size_t count) {
    std::vector<uint32> numbers(count);

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

    return numbers;
}


template <>
std::vector<uint32>
GenerateRandomUint32<ExecutionPolicy::Parallel>(const uint64 key, const size_t count) {
    std::vector<uint32> numbers(count);

    const auto idxes = std::views::iota(uint64{0}, uint64{count / 32});
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 32;
            std::array<uint64, 8> counters{};
            for (size_t j = 0; j < 8; ++j) {
                counters[j] = (i * 8 + j);
            }
            GeneratePhilox8x4x32_10(key, counters, &numbers[offset]);
        }
    );
    const size_t rem = count % 32;
    if (rem > 0) {
        const auto offset = count - rem;
        std::array<uint64, 8> counters{};
        for (size_t j = 0; j < 8; ++j) {
            counters[j] = (count / 32) * 8 + j;
        }
        GeneratePhilox8x4x32_10(key, counters, &numbers[offset]);
    }
    return numbers;
}


template <>
std::vector<uint64>
GenerateRandomUint64<ExecutionPolicy::Sequential>(const uint64 key, const size_t count) {
    std::vector<uint64> numbers(count);

    const auto Transform = [] (uint32 hi, uint32 lo) -> uint64 {
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

    return numbers;
}


template <>
std::vector<uint64>
GenerateRandomUint64<ExecutionPolicy::Parallel>(const uint64 key, const size_t count) {
    std::vector<uint64> numbers(count);

    const auto idxes = std::views::iota(uint64{0}, uint64{count / 16});
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (uint64 i) {
            const auto offset = i * 16;
            std::array<uint64, 8> counters{};
            for (size_t j = 0; j < 8; ++j) {
                counters[j] = (i * 8 + j);
            }
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
        std::array<uint64, 8> counters{};
        for (size_t j = 0; j < 8; ++j) {
            counters[j] = (count / 16) * 8 + j;
        }
        std::array<uint32, 32> numbers_;
        GeneratePhilox8x4x32_10(key, counters, numbers_.data());
        for (size_t i = 0; i < rem; ++i) {
            numbers[offset + i] = (static_cast<uint64>(numbers_[i * 2]) << 32) | static_cast<uint64>(numbers_[i * 2 + 1]);
        }
    }
    return numbers;
}


template <>
std::vector<double>
GenerateRandomContinuous<ExecutionPolicy::Sequential>(const uint64 key, const size_t count) {
    // Generate raw 64-bit randoms first
    std::vector<uint64> uint64_numbers = GenerateRandomUint64<ExecutionPolicy::Sequential>(key, count);

    std::vector<double> numbers(count);

    // constants
    constexpr double INV2P53 = 1.0 / static_cast<double>(1ull << 53);

    // Transform each 64-bit random to double in [0,1)
    std::for_each(std::execution::seq,
        std::views::iota(size_t{0}, count).begin(),
        std::views::iota(size_t{0}, count).end(),
        [&](size_t i) {
            // Take top 53 bits for mantissa
            const uint64 mantissa = uint64_numbers[i] >> (64 - 53);
            numbers[i] = static_cast<double>(mantissa) * INV2P53;
        }
    );

    return numbers;
}


template <>
std::vector<double>
GenerateRandomContinuous<ExecutionPolicy::Parallel>(const uint64 key, const size_t count) {
    // Generate raw 64-bit randoms first
    std::vector<uint64> uint64_numbers = GenerateRandomUint64<ExecutionPolicy::Parallel>(key, count);

    std::vector<double> numbers(count);

    // constants
    constexpr double INV2P53 = 1.0 / static_cast<double>(1ull << 53);

    // Transform each 64-bit random to double in [0,1)
    std::for_each(std::execution::par,
        std::views::iota(size_t{0}, count).begin(),
        std::views::iota(size_t{0}, count).end(),
        [&](size_t i) {
            // Take top 53 bits for mantissa
            const uint64 mantissa = uint64_numbers[i] >> (64 - 53);
            numbers[i] = static_cast<double>(mantissa) * INV2P53;
        }
    );

    return numbers;
}


template <>
std::vector<uint32>
GenerateRandomDiscrete<ExecutionPolicy::Sequential>(const uint64 key, const size_t count, const uint32 max) {
    std::vector<uint32> uint32_numbers = GenerateRandomUint32<ExecutionPolicy::Sequential>(key, count);
    std::vector<uint32> numbers(count);
    std::for_each(std::execution::seq,
        std::views::iota(size_t{0}, count).begin(),
        std::views::iota(size_t{0}, count).end(),
        [&](size_t i) {
            numbers[i] = static_cast<uint32>(static_cast<uint64>(uint32_numbers[i]) * static_cast<uint64>(max) >> 32);
        }
    );
    return numbers;
}

template <>
std::vector<uint32>
GenerateRandomDiscrete<ExecutionPolicy::Parallel>(const uint64 key, const size_t count, const uint32 max) {
    std::vector<uint32> uint32_numbers = GenerateRandomUint32<ExecutionPolicy::Parallel>(key, count);
    std::vector<uint32> numbers(count);
    std::for_each(std::execution::par,
        std::views::iota(size_t{0}, count).begin(),
        std::views::iota(size_t{0}, count).end(),
        [&](size_t i) {
            numbers[i] = static_cast<uint32>(static_cast<uint64>(uint32_numbers[i]) * static_cast<uint64>(max) >> 32);
        }
    );
    return numbers;
}
