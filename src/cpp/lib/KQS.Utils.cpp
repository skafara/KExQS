#include "KQS.Utils.hpp"

#include <execution>
#include <atomic>


template <>
void
FlushSamples<ExecutionPolicy::Sequential>(const std::span<uint> StateCounts, const std::vector<uint> &samples) {
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint sample) {
            StateCounts[sample]++;
        }
    );
}


template <>
void
FlushSamples<ExecutionPolicy::Parallel>(const std::span<uint> StateCounts, const std::vector<uint> &samples) {
    std::for_each(std::execution::par, samples.begin(), samples.end(),
        [&](uint sample) {
            std::atomic_ref<uint>(StateCounts[sample]).fetch_add(1, std::memory_order_relaxed);
        }
    ); // TODO Parallel Reduce
}


template <>
void
FlushSamples<ExecutionPolicy::Accelerated>(const std::span<uint> StateCounts, const std::vector<uint> &samples) {
    FlushSamples<ExecutionPolicy::Parallel>(StateCounts, samples);
}
