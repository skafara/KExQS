#pragma once

#include "KQS.Utils.hpp"

#include <utility>
#include <vector>
#include <span>


/**
 * @brief [LEGACY] Represents a complex number with double-precision real and imaginary parts.
 */
typedef struct {
    /** Real part */
    double Re;
    /** Imaginary part */
    double Im;
} LComplex;


/**
 * @brief [WRAPPER] Deinterleaves an array of LComplex numbers into separate arrays for real and imaginary parts.
 * @param input The input array of LComplex numbers.
 * @return A pair of AlignedVector64<double> containing the real parts and imaginary parts respectively.
 */
template <ExecutionPolicy Policy>
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex(std::span<const LComplex> input);

/**
 * @brief [DELEGATE] Deinterleaves an array of LComplex numbers into separate arrays for real and imaginary parts.
 * @param input The input array of LComplex numbers.
 * @param res The output array for real parts.
 * @param ims The output array for imaginary parts.
 */
template <ExecutionPolicy Policy>
inline
void
_DeinterleaveAoSLComplex(std::span<const LComplex> input, std::span<double> res, std::span<double> ims);


/**
 * @brief Calculates the probability from real and imaginary parts of a complex number.
 * @param re The real part.
 * @param im The imaginary part.
 * @return The calculated probability.
 */
inline
double
CalculateProbability(const double re, const double im);

/**
 * @brief Calculates the probabilities from real and imaginary parts of a complex numbers using manual AVX2 intrinsics.
 * @param re The input __m256d register containing real parts.
 * @param im The input __m256d register containing imaginary parts.
 * @return An __m256d register containing the calculated probabilities.
 */
inline
__m256d
CalculateProbability(const __m256d re, const __m256d im);


/**
 * @brief [DELEGATE] Calculates the probabilities from arrays of real and imaginary parts of complex numbers.
 * @param res The input array of real parts.
 * @param ims The input array of imaginary parts.
 * @param probs The output array for calculated probabilities.
 */
template <ExecutionPolicy Policy>
inline
void
_CalculateProbabilities(typename DeviceContainer<Policy, double>::ref_const_type res, typename DeviceContainer<Policy, double>::ref_const_type ims, std::span<double> probs);

/**
 * @brief [WRAPPER] Calculates the probabilities from arrays of real and imaginary parts of complex numbers.
 * @param res The input array of real parts.
 * @param ims The input array of imaginary parts.
 * @return An AlignedVector64<double> containing the calculated probabilities.
 */
template <ExecutionPolicy Policy>
AlignedVector64<double>
CalculateProbabilities(std::span<const double> res, std::span<const double> ims);
