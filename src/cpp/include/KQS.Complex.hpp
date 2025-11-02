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
_DeinterleaveAoSLComplex(const std::span<const LComplex> arr, std::vector<double>& res, std::vector<double>& ims);
