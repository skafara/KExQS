#include "KQS.Complex.hpp"


using uint = unsigned int;

    
extern "C" __declspec(dllexport) void __cdecl ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumStates,
    uint ANumShots
);


template <ExecutionPolicy Policy>
void
Run(const std::span<uint> AStateCounts, const std::span<const LComplex> AStateAmplitudes, const uint ANumShots);


template <ExecutionPolicy Policy>
std::vector<double>
CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims);

template <ExecutionPolicy Policy>
void
_CalculateProbabilities(const std::vector<double> &res, const std::vector<double> &ims, std::vector<double> &probs);


template <ExecutionPolicy Policy>
void
FlushSamples(std::span<uint> counts, const std::vector<uint32_t> &samples);
