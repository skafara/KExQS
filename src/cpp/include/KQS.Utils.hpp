#pragma once

#include <cstdint>
#include <new>
#include <memory>
#include <span>
#include <vector>
#include <chrono>
#include <CL/opencl.hpp>


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
void
FlushSamples(std::span<uint> StateCounts, std::span<uint> samples);
