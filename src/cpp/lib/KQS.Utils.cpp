#include "KQS.Utils.hpp"

#include <execution>
#include <atomic>


template <>
void
FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> StateCounts, const std::vector<uint> &samples) {
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint sample) {
            StateCounts[sample]++;
        }
    );
}


template <>
void
FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> StateCounts, const std::vector<uint32_t> &samples) {
    std::for_each(std::execution::par, samples.begin(), samples.end(),
        [&](uint32_t sample) {
            // atomic increment to avoid race conditions
            std::atomic_ref<uint>(StateCounts[sample]).fetch_add(1, std::memory_order_relaxed);
        }
    ); // TODO without atomic operations using thread-local StateCounts and then reduce
}


template <>
void
FlushSamples<ExecutionPolicy::Accelerated>(std::span<uint> StateCounts, const std::vector<uint> &samples) {
    FlushSamples<ExecutionPolicy::Parallel>(StateCounts, samples);
}
