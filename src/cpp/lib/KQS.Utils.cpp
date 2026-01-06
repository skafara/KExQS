#include "KQS.Utils.hpp"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <execution>
#include <atomic>


template <>
inline
void
_FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> StateCounts, std::span<uint> samples) {
    // [Sequential]
    std::for_each(std::execution::seq, samples.begin(), samples.end(),
        [&] (uint sample) {
            StateCounts[sample]++;
        }
    );
}

template <>
inline
void
_FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> StateCounts, std::span<uint> samples) {
    // [Parallel] Sort samples
    std::sort(std::execution::par_unseq, samples.begin(), samples.end());

    /**
     * @brief Run structure for parallel reduction
     */
    struct Run {
        uint value;
        size_t count;
    };

    // [Parallel] Reduce to runs
    const std::vector<Run> result = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>(0, samples.size()),
        // Identity
        std::vector<Run>{},
        // Process range
        [&] (const tbb::blocked_range<size_t> &range, std::vector<Run> local) {
            if (range.empty()) {
                // Return already processed runs
                return local;
            }

            auto it = range.begin();
            const auto end = range.end();

            if (samples[it] == samples[end - 1]) {
                // All samples in range are the same
                if (local.size() > 0 && local.back().value == samples[it]) {
                    // If there is a previous run with the same value, merge counts
                    local.back().count += end - it;
                    return local;
                }
                
                // Otherwise, add new run
                local.emplace_back(samples[it], end - it);
                return local;
            }

            // General case: iterate through samples in range
            auto it_ = it;
            ++it;
            for (; it < end; ++it) {
                if (samples[it] != samples[it_]) {
                    // Different value encountered
                    if (local.size() > 0 && local.back().value == samples[it_]) {
                        // If there is a previous run with the same value, merge counts
                        local.back().count += it - it_;
                        // Update iterator
                        it_ = it;
                        // And continue to next sample
                        continue;
                    }
                    
                    // Add new run
                    local.emplace_back(samples[it_], it - it_);
                    // Update iterator
                    it_ = it;
                }
            }
            
            // Handle last run
            local.emplace_back(samples[end - 1], it - it_);
            return local;
        },
        // Combine results
        [&] (std::vector<Run> lhs, const std::vector<Run> &rhs) {
            if (rhs.empty()) {
                return lhs;
            }

            if (lhs.empty()) {
                return rhs;
            }

            if (lhs.back().value == rhs.front().value) {
                // If the last run in lhs has the same value as the first in rhs, merge counts
                lhs.back().count += rhs.front().count;
                lhs.insert(lhs.end(), rhs.begin() + 1, rhs.end());
            } else {
                // Otherwise, just concatenate
                lhs.insert(lhs.end(), rhs.begin(), rhs.end());
            }
            return lhs;
        }
    );

    // [Parallel]
    std::for_each(std::execution::par, result.begin(), result.end(),
        [&] (const Run& run) {
            StateCounts[run.value] = run.count;
        }
    );
}


template <ExecutionPolicy Policy>
void
FlushSamples(std::span<uint> StateCounts, std::span<uint> samples) {
    // [BENCHMARK] _FlushSamples
    BenchmarkedFuncRun("_FlushSamples",
        [&] () {
            // [DELEGATE] Call
            _FlushSamples<Policy>(StateCounts, samples);
        }
    );
}

template
void
FlushSamples<ExecutionPolicy::Sequential>(std::span<uint> StateCounts, std::span<uint> samples);

template
void
FlushSamples<ExecutionPolicy::Parallel>(std::span<uint> StateCounts, std::span<uint> samples);

template <>
void
FlushSamples<ExecutionPolicy::Accelerated>(std::span<uint> StateCounts, std::span<uint> samples) {
    // [FALLBACK] [Parallel]
    FlushSamples<ExecutionPolicy::Parallel>(StateCounts, samples);
}


BenchmarkRegistry &BenchmarkRegistry::Instance() {
    static BenchmarkRegistry instance;
    return instance;
}

void BenchmarkRegistry::Record(const std::string &name, std::chrono::nanoseconds duration) {
    _benchmarks[name].push_back(duration);
}

BenchmarkRegistry::Result BenchmarkRegistry::GetResult(const std::string &name) {
    const auto &durations = _benchmarks[name];
    if (durations.empty()) {
        throw std::runtime_error("No benchmark data for " + name);
    }
    
    double min = static_cast<double>(durations[0].count());
    double max = static_cast<double>(durations[0].count());
    double sum = 0.0;
    for (const auto &dur : durations) {
        const double dur_ = static_cast<double>(dur.count());
        if (dur_ < min) {
            min = dur_;
        }
        if (dur_ > max) {
            max = dur_;
        }
        sum += dur_;
    }
    const double mean = sum / static_cast<double>(durations.size());

    double variance_sum = 0.0;
    for (const auto &dur : durations) {
        const double dur_ = static_cast<double>(dur.count());
        variance_sum += (dur_ - mean) * (dur_ - mean);
    }
    const double variance = variance_sum / static_cast<double>(durations.size() - 1);
    const double stddev = std::sqrt(variance);
    const double ci95 = 1.96 * stddev / std::sqrt(static_cast<double>(durations.size()));

    return {min, max, mean, ci95};
}

void BenchmarkRegistry::Clear() {
    _benchmarks.clear();
}


ScopedTimer::ScopedTimer(const std::string &name) : _name(name), _start(clock_type::now()) {
    // Start timer
}

ScopedTimer::~ScopedTimer() {
    // Stop timer
    const auto end = clock_type::now();
    const auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - _start);
    // Record duration
    BenchmarkRegistry::Instance().Record(_name, dur);
}
