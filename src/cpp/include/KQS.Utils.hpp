#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <CL/opencl.hpp>


using uint = unsigned int;
using uint32 = uint32_t;
using uint64 = uint64_t;


enum class ExecutionPolicy {
    Sequential,
    Parallel,
    Accelerated
};


template <ExecutionPolicy Policy>
struct DeviceContainer {
    using type = std::vector<double>;
};

template <>
struct DeviceContainer<ExecutionPolicy::Accelerated> {
    using type = cl::Buffer;
    //using type = std::vector<double>;
};


template <ExecutionPolicy Policy>
void
FlushSamples(const std::span<uint> StateCounts, std::vector<uint> &samples);
