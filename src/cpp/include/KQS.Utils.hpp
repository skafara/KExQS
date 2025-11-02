#pragma once

#include <cstdint>


using uint = unsigned int;
using uint32 = uint32_t;
using uint64 = uint64_t;


enum class ExecutionPolicy {
    Sequential,
    Parallel,
    Accelerated
};
