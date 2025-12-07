#pragma once

#include "KQS.Utils.hpp"

#include <utility>
#include <vector>
#include <span>


typedef struct {
    double Re;
    double Im;
} LComplex;


template <ExecutionPolicy Policy>
std::pair<AlignedVector64<double>, AlignedVector64<double>>
DeinterleaveAoSLComplex(std::span<const LComplex> arr);

template <ExecutionPolicy Policy>
inline
void
_DeinterleaveAoSLComplex(std::span<const LComplex> arr, std::span<double> res, std::span<double> ims);


inline
double
CalculateProbability(const double re, const double im);

inline
__m256d
CalculateProbability(const __m256d re, const __m256d im);


template <ExecutionPolicy Policy>
AlignedVector64<double>
CalculateProbabilities(std::span<const double> res, std::span<const double> ims);

template <ExecutionPolicy Policy>
inline
void
_CalculateProbabilities(std::span<const double> res, std::span<const double> ims, std::span<double> probs);
