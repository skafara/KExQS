#include "KQS.Utils.hpp"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <execution>
#include <atomic>


template <>
void
FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> StateCounts, std::span<uint> samples) {
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint sample) {
            StateCounts[sample]++;
        }
    );
}

template <>
void
FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> StateCounts, std::span<uint> samples) {
    std::sort(std::execution::par_unseq, samples.begin(), samples.end());

    struct Run {
        uint value;
        size_t count;
    };

    const std::vector<Run> result = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, samples.size()),
        std::vector<Run>{},
        [&] (const tbb::blocked_range<size_t> &range, std::vector<Run> local) {
            if (range.empty()) {
                return local;
            }

            auto it = range.begin();
            const auto end = range.end();

            if (samples[it] == samples[end - 1]) {
                if (local.size() > 0 && local.back().value == samples[it]) {
                    local.back().count += end - it;
                    return local;
                }
                
                local.emplace_back(samples[it], end - it);
                return local;
            }

            auto it_ = it;
            ++it;
            for (; it < end; ++it) {
                if (samples[it] != samples[it_]) {
                    if (local.size() > 0 && local.back().value == samples[it_]) {
                        local.back().count += it - it_;
                        it_ = it;
                        continue;
                    }
                    
                    local.emplace_back(samples[it_], it - it_);
                    it_ = it;
                }
            }
            
            local.emplace_back(samples[end - 1], it - it_);
            return local;
        },
        [&] (std::vector<Run> lhs, const std::vector<Run> &rhs) {
            if (rhs.empty()) {
                return lhs;
            }

            if (lhs.empty()) {
                return rhs;
            }

            if (lhs.back().value == rhs.front().value) {
                lhs.back().count += rhs.front().count;
                lhs.insert(lhs.end(), rhs.begin() + 1, rhs.end());
            } else {
                lhs.insert(lhs.end(), rhs.begin(), rhs.end());
            }
            return lhs;
        }
    );

    std::for_each(std::execution::par, result.begin(), result.end(),
        [&] (const Run& run) {
            StateCounts[run.value] = run.count;
        }
    );
}


template <>
void
FlushSamples<ExecutionPolicy::Accelerated>(std::span<uint> StateCounts, std::span<uint> samples) {
    FlushSamples<ExecutionPolicy::Parallel>(StateCounts, samples);
}
