#include "KQS.Simulator.hpp"
#include <stdio.h>


void ESimulator_Run(
    uint* AStateCounts,
    const LComplex* AStateAmplitudes,
    uint ANumQubits,
    uint ANumStates,
    uint ANumShots
) {
    for (size_t i = 0; i < ANumStates; i++) {
        AStateCounts[i] = i;
        printf("State %zu: Amplitude = %.6f + %.6fi\n", i, AStateAmplitudes[i].Re, AStateAmplitudes[i].Im);
    }
}
