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

/** @brief Cache line size in bytes */
constexpr size_t CacheLineSize = 64;

/** @brief Whether benchmarking is enabled */
constexpr bool BenchmarkingEnabled = BENCHMARKING_ENABLED;
/** @brief Number of warmup iterations for benchmarking */
constexpr size_t BenchmarkingWarmupIterations = 10;
/** @brief Number of measured iterations for benchmarking */
constexpr size_t BenchmarkingMeasuredIterations = 100;


using uint = unsigned int;
using uint32 = uint32_t;
using uint64 = uint64_t;

using clock_type = std::chrono::high_resolution_clock;


/** @brief Execution policy for parallelism */
enum class ExecutionPolicy {
    /** @brief Sequential execution */
    Sequential,
    /** @brief Parallel + SIMD execution on CPU */
    Parallel,
    /** @brief OpenCL accelerated execution on GPU */
    Accelerated
};


/** @brief Pseudorandom number generator algorithm */
enum class PrngAlgorithm {
    /** @brief Philox algorithm */
    Philox,
    /** @brief Mersenne Twister 19937 algorithm */
    MT19937,
    /** @brief Random.org historical random data */
    RandomOrg
};


/**
 * @brief Aligned allocator for STL containers
 * @tparam T Type of elements
 * @tparam Alignment Alignment in bytes
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    /**
     * @brief Rebind allocator to another type
     * @tparam U New type
     * @return AlignedAllocator<U, Alignment>
     */
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    /** @brief Default constructor */
    AlignedAllocator() noexcept = default;

    /**
     * @brief Copy constructor from another aligned allocator
     * @tparam U Type of other allocator
     * @param other Other allocator
     * @return AlignedAllocator<T, Alignment>
     */
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    /**
     * @brief Allocate aligned memory
     * @param n Number of elements
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    value_type* allocate(size_t n) {
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return static_cast<value_type*>(ptr);
    }

    /**
     * @brief Deallocate aligned memory
     * @param p Pointer to memory
     * @param n Number of elements
     */
    void deallocate(value_type* p, size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }
};


/**
 * @brief Aligned vector using AlignedAllocator
 * @tparam T Type of elements
 * @tparam Alignment Alignment in bytes
 */
template <typename T, std::size_t Alignment>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

/**
 * @brief Aligned vector with 64-byte alignment
 * @tparam T Type of elements
 */
template <typename T>
using AlignedVector64 = AlignedVector<T, 64>;


/**
 * @brief Device container mapping based on execution policy
 * @tparam Policy Execution policy
 * @tparam T Type of elements
 * @return Corresponding container type
 */
template <ExecutionPolicy Policy, typename T>
struct DeviceContainer {
    /** @brief Type of container */
    using type = AlignedVector64<T>;
    /** @brief Reference type */
    using ref_type = std::span<T>;
    /** @brief Constant reference type */
    using ref_const_type = std::span<const T>;
};

/**
 * @brief Specialization of DeviceContainer for Accelerated policy
 * @tparam T Type of elements
 * @return cl::Buffer type
 */
template <typename T>
struct DeviceContainer<ExecutionPolicy::Accelerated, T> {
    using type = cl::Buffer;
    using ref_type = type &;
    using ref_const_type = const type &;
};


/**
 * @brief [DELEGATE] Flush samples to state counts
 * @param StateCounts Output state counts
 * @param samples Input samples
 * @return void
 */
template <ExecutionPolicy Policy>
inline
void
_FlushSamples(std::span<uint> StateCounts, std::span<uint> samples);

/**
 * @brief [WRAPPER] Flush samples to state counts
 * @param StateCounts Output state counts
 * @param samples Input samples
 * @return void
 */
template <ExecutionPolicy Policy>
void
FlushSamples(std::span<uint> StateCounts, std::span<uint> samples);


/**
 * @brief [BENCHMARK] Registry for benchmark results
 */
class BenchmarkRegistry {
public:
    /** @brief Result structure for benchmark statistics */
    struct Result {
        /** @brief Minimum duration */
        double Min;
        /** @brief Maximum duration */
        double Max;
        /** @brief Mean duration */
        double Mean;
        /** @brief Confidence interval at 95% */
        double CI95;
    };

    /**
     * @brief Get singleton instance of BenchmarkRegistry
     * @return Reference to BenchmarkRegistry instance
     */
    static BenchmarkRegistry &Instance();

    /**
     * @brief Record a benchmark duration
     * @param name Name of the benchmark
     * @param duration Duration to record
     */
    void Record(const std::string &name, std::chrono::nanoseconds duration);

    /**
     * @brief Get benchmark result statistics
     * @param name Name of the benchmark
     * @return Result structure containing benchmark statistics
     * @throws std::runtime_error if no benchmark data is available
     */
    Result GetResult(const std::string &name);
    
    /**
     * @brief Clear all recorded benchmark data
     */
    void Clear();

private:
    BenchmarkRegistry() = default;
    ~BenchmarkRegistry() = default;

    /** @brief Map of benchmark names to recorded durations */
    std::map<std::string, std::vector<std::chrono::nanoseconds>> _benchmarks;
};


/**
 * @brief Scoped timer for benchmarking
 */
class ScopedTimer {
public:
    /**
     * @brief Constructor - starts the timer
     * @param name Name of the benchmark
     */
    ScopedTimer(const std::string &name);
    /**
     * @brief Destructor - stops the timer and records the duration into the registry
     */
    ~ScopedTimer();

private:
    std::string _name;
    const clock_type::time_point _start;
};


/**
 * @brief Run a function - with benchmarking if enabled
 * @param name Name of the benchmark
 * @param func Function to run
 */
inline
void
BenchmarkedFuncRun(const std::string &name, const std::function<void()> &func) {
    if constexpr (!BenchmarkingEnabled) {
        // No benchmarking
        func();
    }
    else {
        // Warmup iterations
        for (size_t i = 0; i < BenchmarkingWarmupIterations; ++i) {
            func();
        }

        // Measured iterations
        for (size_t i = 0; i < BenchmarkingMeasuredIterations; ++i) {
            // Scoped timer function execution
            ScopedTimer timer(name);
            func();
        }
    }
}

/**
 * @brief Run a kernel function - with benchmarking if enabled
 * @param name Name of the benchmark
 * @param func Function returning bound cl::Event used to measure kernel execution time
 */
inline
void
BenchmarkedKernelRun(const std::string &name, const std::function<cl::Event()> &func) {
    if constexpr (!BenchmarkingEnabled) {
        // No benchmarking
        func();
    }
    else {
        // Warmup iterations
        for (size_t i = 0; i < BenchmarkingWarmupIterations; ++i) {
            func();
        }

        // Measured iterations
        for (size_t i = 0; i < BenchmarkingMeasuredIterations; ++i) {
            // Execute kernel and get event
            auto event = func();
            // Wait for completion
            event.wait();
            // Retrieve profiling info
            const auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            const auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            // Record duration manually
            BenchmarkRegistry::Instance().Record(name, std::chrono::nanoseconds(end - start));
        }
    }
}
