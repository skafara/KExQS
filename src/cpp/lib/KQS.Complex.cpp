#include "KQS.Utils.hpp"
#include "KQS.Complex.hpp"
#include "KQS.CLManager.hpp"

#include <execution>
#include <ranges>
#include <immintrin.h>


template <ExecutionPolicy Policy>
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex(std::span<const LComplex> arr) {
    static_assert(std::is_standard_layout_v<LComplex> && sizeof(LComplex) == 16);

    AlignedVector64<double> res(arr.size()); // TODO align to 64 bytes
    AlignedVector64<double> ims(arr.size()); // TODO align to 64 bytes

    _DeinterleaveAoSLComplex<Policy>(arr, res, ims);
    return {res, ims};
}


template
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Sequential>(std::span<const LComplex> arr);


template
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(std::span<const LComplex> arr);


template
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex<ExecutionPolicy::Accelerated>(std::span<const LComplex> arr);


template <>
inline
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Sequential>(std::span<const LComplex> arr, std::span<double> res, std::span<double> ims) {
    const auto idxes = std::views::iota(size_t{0}, arr.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            res[i] = arr[i].Re;
            ims[i] = arr[i].Im;
        }
    );
}

template <>
inline
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(std::span<const LComplex> arr, std::span<double> res, std::span<double> ims) {
    const auto idxes = std::views::iota(size_t{0}, arr.size() / 4) | std::views::transform([] (size_t i) { return i * 4; });
    // 1 Block = 4 Complex = 8 doubles = 2x 256-bit AVX2 registers
    // [Re0, Im0, Re1, Im1, Re2, Im2, Re3, Im3]
    std::for_each(std::execution::par, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            const __m256d c12 = _mm256_load_pd(reinterpret_cast<const double*>(&arr[i])); // [Re0, Im0, Re1, Im1]
            const __m256d c34 = _mm256_load_pd(reinterpret_cast<const double*>(&arr[i + 2])); // [Re2, Im2, Re3, Im3]
            __m256d re_ = _mm256_shuffle_pd(c12, c34, 0b0000); // [Re0, Re2, Re1, Re3]
            __m256d im_ = _mm256_shuffle_pd(c12, c34, 0b1111); // [Im0, Im2, Im1, Im3]
            re_ = _mm256_permute4x64_pd(re_, 0b11011000); // 0, 2, 1, 3 -> [Re0, Re1, Re2, Re3]
            im_ = _mm256_permute4x64_pd(im_, 0b11011000); // 0, 2, 1, 3 -> [Im0, Im1, Im2, Im3]
            _mm256_store_pd(&res[i], re_);
            _mm256_store_pd(&ims[i], im_);
        }
    );

    const size_t rem = arr.size() % 4;
    if (rem > 0) {
        const auto offset = arr.size() - rem;
        const auto idxes = std::views::iota(size_t{offset}, arr.size());
        std::for_each(std::execution::par, idxes.begin(), idxes.end(),
            [&] (size_t i) {
                res[i] = arr[i].Re;
                ims[i] = arr[i].Im;
            }
        );
    }
}

template <>
inline
void
_DeinterleaveAoSLComplex<ExecutionPolicy::Accelerated>(std::span<const LComplex> arr, std::span<double> res, std::span<double> ims) {
    _DeinterleaveAoSLComplex<ExecutionPolicy::Parallel>(arr, res, ims);
}


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
AlignedVector64<double>
CalculateProbabilities(std::span<const double> res, std::span<const double> ims) {
    AlignedVector64<double> probs(res.size());
    _CalculateProbabilities<Policy>(res, ims, probs);
    return probs;
}

template
AlignedVector64<double>
CalculateProbabilities<ExecutionPolicy::Sequential>(std::span<const double> res, std::span<const double> ims);

template
AlignedVector64<double>
CalculateProbabilities<ExecutionPolicy::Parallel>(std::span<const double> res, std::span<const double> ims);

template <>
AlignedVector64<double>
CalculateProbabilities<ExecutionPolicy::Accelerated>(std::span<const double> res, std::span<const double> ims) {
    AlignedVector64<double> probs(res.size());

    CLManager &clManager = CLManager::Instance();
    cl::Kernel &kernel = clManager.GetKernel("_CalculateProbabilities");
    
    const size_t dataSize = res.size() * sizeof(double);
    
    cl::Buffer resBuffer(clManager.GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, const_cast<double*>(res.data()));
    cl::Buffer imsBuffer(clManager.GetContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, const_cast<double*>(ims.data()));
    cl::Buffer probsBuffer(clManager.GetContext(), CL_MEM_WRITE_ONLY, dataSize);

    kernel.setArg(0, resBuffer);
    kernel.setArg(1, imsBuffer);
    kernel.setArg(2, probsBuffer);

    clManager.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res.size()), cl::NullRange);
    clManager.GetCommandQueue().enqueueReadBuffer(probsBuffer, CL_TRUE, 0, dataSize, probs.data());
    
    clManager.GetCommandQueue().finish();
    return probs;
}


template <>
inline
void
_CalculateProbabilities<ExecutionPolicy::Sequential>(std::span<const double> res, std::span<const double> ims, std::span<double> probs) {
    const auto idxes = std::views::iota(size_t{0}, res.size());
    std::for_each(std::execution::seq, idxes.begin(), idxes.end(),
        [&] (size_t i) {
            probs[i] = CalculateProbability(res[i], ims[i]);
        }
    );
}


template <>
inline
void
_CalculateProbabilities<ExecutionPolicy::Parallel>(std::span<const double> res, std::span<const double> ims, std::span<double> probs) {
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
