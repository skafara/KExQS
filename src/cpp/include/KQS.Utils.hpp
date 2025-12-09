#pragma once

#include <cstdint>
#include <new>
#include <memory>
#include <span>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <CL/opencl.hpp>


#ifndef BENCHMARKING_ENABLED
#define BENCHMARKING_ENABLED false
#else
#undef BENCHMARKING_ENABLED
#define BENCHMARKING_ENABLED true
#endif


constexpr bool BenchmarkingEnabled = BENCHMARKING_ENABLED;
constexpr size_t BenchmarkingWarmupIterations = 1;
constexpr size_t BenchmarkingMeasuredIterations = 10;


using uint = unsigned int;
using uint32 = uint32_t;
using uint64 = uint64_t;

using clock_type = std::chrono::high_resolution_clock;


enum class ExecutionPolicy {
    Sequential,
    Parallel,
    Accelerated
};


enum class PrngAlgorithm {
    Philox,
    MT19937,
    RandomOrg
};


template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    value_type* allocate(size_t n) {
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return static_cast<value_type*>(ptr);
    }

    void deallocate(value_type* p, size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }
};

template <typename T, std::size_t Alignment>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

template <typename T>
using AlignedVector64 = AlignedVector<T, 64>;


template <ExecutionPolicy Policy, typename T>
struct DeviceContainer {
    using type = AlignedVector64<T>;
    using ref_type = std::span<T>;
    using const_ref_type = std::span<const T>;
};

template <typename T>
struct DeviceContainer<ExecutionPolicy::Accelerated, T> {
    using type = cl::Buffer;
    using ref_type = type;
    using const_ref_type = const type;
};


template <ExecutionPolicy Policy>
inline
void
_FlushSamples(std::span<uint> StateCounts, std::span<uint> samples);

template <ExecutionPolicy Policy>
void
FlushSamples(std::span<uint> StateCounts, std::span<uint> samples);


class BenchmarkRegistry {
public:
    struct Result {
        double Min;
        double Max;
        double Mean;
        double CI95;
    };

    static BenchmarkRegistry &Instance();
    void Record(const std::string &name, std::chrono::nanoseconds duration);
    Result GetResult(const std::string &name);

private:
    BenchmarkRegistry() = default;
    ~BenchmarkRegistry() = default;

    std::map<std::string, std::vector<std::chrono::nanoseconds>> _benchmarks;
};


class ScopedTimer {
public:
    ScopedTimer(const std::string &name);
    ~ScopedTimer();

private:
    std::string _name;
    const clock_type::time_point _start;
};


inline
void
BenchmarkedFuncRun(const std::string &name, const std::function<void()> &func) {
    if constexpr (!BenchmarkingEnabled) {
        func();
    }
    else {
        for (size_t i = 0; i < BenchmarkingWarmupIterations; ++i) {
            func();
        }

        for (size_t i = 0; i < BenchmarkingMeasuredIterations; ++i) {
            ScopedTimer timer(name);
            func();
        }
    }
}

inline
void
BenchmarkedKernelRun(const std::string &name, const std::function<cl::Event()> &func) {
    if constexpr (!BenchmarkingEnabled) {
        func();
    }
    else {
        for (size_t i = 0; i < BenchmarkingWarmupIterations; ++i) {
            func();
        }

        for (size_t i = 0; i < BenchmarkingMeasuredIterations; ++i) {
            auto event = func();
            event.wait();
            const auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            const auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            BenchmarkRegistry::Instance().Record(name, std::chrono::nanoseconds(end - start));
        }
    }
}
