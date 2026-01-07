typedef uint uint32;
typedef ulong uint64;


/** @brief Inverse of 2^53 - used for continuising uint64 to double */
const double INV2P53 = 1.0 / (double) (1UL << 53);


/**
 * @brief Generates 4x32-bit pseudorandom numbers using Philox 4x32-10 algorithm
 * @param key Key
 * @param counter Counter
 * @param out Output array of 4 uint32 numbers
 */
inline
void
GeneratePhilox4x32_10(const uint64 key, const uint64 counter, __private uint32 out[4]) {
    const uint64 M0 = 0xD2511F53UL;
    const uint64 M1 = 0xCD9E8D57UL;
    const uint32 W0 = 0x9E3779B9UL;
    const uint32 W1 = 0xBB67AE85UL;

    uint32 k0 = (uint32) key;
    uint32 k1 = (uint32) (key >> 32);

    uint32 x0 = (uint32) counter;
    uint32 x1 = (uint32) (counter >> 32);
    uint32 x2 = 0u;
    uint32 x3 = 0u;

    for (int round = 0; round < 10; ++round) {
        const uint64 p0 = M0 * x0;
        const uint64 p1 = M1 * x2;

        const uint32 hi0 = (uint32) (p0 >> 32);
        const uint32 hi1 = (uint32) (p1 >> 32);
        const uint32 lo0 = (uint32) p0;
        const uint32 lo1 = (uint32) p1;

        const uint32 y0 = hi1 ^ x1 ^ k0;
        const uint32 y1 = lo1;
        const uint32 y2 = hi0 ^ x3 ^ k1;
        const uint32 y3 = lo0;

        x0 = y0;
        x1 = y1;
        x2 = y2;
        x3 = y3;

        k0 += W0;
        k1 += W1;
    }

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = x3;
}


/**
 * @brief Discretises a uint32 number to [0, max)
 * @param number Input uint32 number
 * @param max Upper bound (exclusive)
 * @return Discretised uint32 number in [0, max)
 */
inline
uint32
Discretise(const uint32 number, const uint32 max) {
    return (uint32) (((uint64) number * (uint64) max) >> 32);
}

/**
 * @brief [KERNEL] Generates random discrete numbers in [0, max)
 *        [WORK ITEM] 4 uint32 numbers -> out[4*i ... 4*i+3]
 * @param key Key
 * @param max Upper bound (exclusive)
 * @param out Output array of uint32 numbers
 */
__kernel
void
_GenerateRandomDiscrete(const uint64 key, const uint32 max, __global uint32 *out) {
    const size_t i = get_global_id(0);
    
    // Generate
    uint32 numbers[4];
    GeneratePhilox4x32_10(key, (uint64) i, numbers);
    
    // Discretise
    // Write
    for (int j = 0; j < 4; ++j) {
        out[4 * i + j] = Discretise(numbers[j], max);
    }
}


/**
 * @brief Packs two uint32 numbers into one uint64
 * @param hi High 32 bits
 * @param lo Low 32 bits
 * @return Packed uint64 number
 */
inline
uint64
Pack(const uint32 hi, const uint32 lo) {
    return ((uint64) hi << 32) | (uint64) lo;
}

/**
 * @brief Continuises a uint64 number to double in [0, 1)
 * @param number Input uint64 number
 * @return Continuous double number in [0, 1)
 */
inline
double
Continuise(const uint64 number) {
    const uint64 mantissa = number >> 11;
    return (double) mantissa * INV2P53;
}

/**
 * @brief [KERNEL] Generates random continuous numbers in [0, 1)
 *        [WORK ITEM] 2 double numbers -> out[2*i, 2*i+1]
 * @param key Key
 * @param out Output array of double numbers
 */
__kernel
void
_GenerateRandomContinuous(const uint64 key, __global double *out) {
    const size_t i = get_global_id(0);
    
    // Generate 4x32-bit numbers
    uint32 numbers32[4];
    GeneratePhilox4x32_10(key, (uint64) i, numbers32);

    // Pack to 2x64-bit numbers
    uint64 numbers64[2];
    for (int j = 0; j < 2; ++j) {
        numbers64[j] = Pack(numbers32[2 * j], numbers32[2 * j + 1]);
    }
    
    // Continuise
    // Write
    for (int j = 0; j < 2; ++j) {
        out[2 * i + j] = Continuise(numbers64[j]);
    }
}


/**
 * @brief [KERNEL] Samples from alias table
 *        [WORK ITEM] 1 sample -> samples[i]
 * @param probs Probability array
 * @param aliases Alias array
 * @param bins Bin indices array
 * @param rands Random numbers array
 * @param samples Output samples array
 */
__kernel
void 
_SampleAliasTable(__global const double *probs, __global const uint32 *aliases, __global const uint32 *bins, __global const double *rands, __global uint32 *samples) {
    const size_t i = get_global_id(0);
    
    // Branchless sample selection
    const uint32 cond = rands[i] < probs[bins[i]];
    // if rand < prob[bin] then sample = bin else sample = alias[bin]
    samples[i] = aliases[bins[i]] + cond * (bins[i] - aliases[bins[i]]);
}
