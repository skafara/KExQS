#include "KQS.Complex.hpp"

#include <execution>
#include <ranges>
#include <immintrin.h>


template <ExecutionPolicy Policy>
std::pair<std::vector<double>, std::vector<double>>
DeinterleaveAoSLComplex(const std::span<const LComplex> arr) {
    static_assert(std::is_standard_layout_v<LComplex> && sizeof(LComplex) == 16);

    std::vector<double> res(arr.size()); // TODO align to 32 bytes
    std::vector<double> ims(arr.size()); // TODO align to 32 bytes

    if (arr.size() == 2) {
        _DeinterleaveAoSLComplex<ExecutionPolicy::Sequential>(arr, res, ims);
    } else {
        _DeinterleaveAoSLComplex<Policy>(arr, res, ims);
    }

    return {res, ims};
}


template
std::pair<std::vector<double>, std::vector<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Sequential>(const std::span<const LComplex> arr);


template
std::pair<std::vector<double>, std::vector<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(const std::span<const LComplex> arr);


template
std::pair<std::vector<double>, std::vector<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Accelerated>(const std::span<const LComplex> arr);


template <>
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Sequential>(const std::span<const LComplex> arr, std::vector<double>& res, std::vector<double>& ims) {
    const auto idxes = std::views::iota(size_t{0}, arr.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            res[i] = arr[i].Re;
            ims[i] = arr[i].Im;
        }
    );
}


template <>
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(const std::span<const LComplex> arr, std::vector<double>& res, std::vector<double>& ims) {
    const auto idxes = std::views::iota(size_t{0}, arr.size() / 4) | std::views::transform([] (size_t i) { return i * 4; });
    // 1 Block = 4 Complex = 8 doubles = 2x 256-bit AVX2 registers
    // [Re0, Im0, Re1, Im1, Re2, Im2, Re3, Im3]
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            __m256d c12 = _mm256_load_pd(reinterpret_cast<const double*>(&arr[i])); // [Re0, Im0, Re1, Im1]
            __m256d c34 = _mm256_load_pd(reinterpret_cast<const double*>(&arr[i + 2])); // [Re2, Im2, Re3, Im3]
            __m256d re_ = _mm256_shuffle_pd(c12, c34, 0b0000); // [Re0, Re2, Re1, Re3]
            __m256d im_ = _mm256_shuffle_pd(c12, c34, 0b1111); // [Im0, Im2, Im1, Im3]
            re_ = _mm256_permute4x64_pd(re_, 0b11011000); // 0, 2, 1, 3 -> [Re0, Re1, Re2, Re3]
            im_ = _mm256_permute4x64_pd(im_, 0b11011000); // 0, 2, 1, 3 -> [Im0, Im1, Im2, Im3]
            _mm256_store_pd(&res[i], re_);
            _mm256_store_pd(&ims[i], im_);
        }
    );
}


template <>
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Accelerated>(const std::span<const LComplex> arr, std::vector<double>& res, std::vector<double>& ims) {
    _DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(arr, res, ims);
}
