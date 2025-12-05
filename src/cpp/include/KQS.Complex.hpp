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
std::pair<std::vector<double>, std::vector<double>>
DeinterleaveAoSLComplex(const std::span<const LComplex> arr);

template <ExecutionPolicy Policy>
void
_DeinterleaveAoSLComplex(const std::span<const LComplex> arr, std::vector<double> &res, std::vector<double> &ims);


inline
double
CalculateProbability(const double re, const double im);

inline
__m256d
CalculateProbability(const __m256d re, const __m256d im);


template <ExecutionPolicy Policy>
std::vector<double>
CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims);

template <ExecutionPolicy Policy>
void
_CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs);
