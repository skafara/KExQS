/**
 * @brief Calculates the probability (squared magnitude) given real and imaginary part of a complex number
 * @param re Real part
 * @param im Imaginary part
 * @return Probability (squared magnitude)
 */
inline
double
CalculateProbability(const double re, const double im) {
    // p = re^2 + im^2
    return re * re + im * im;
}

/**
 * @brief [KERNEL] Calculates the probabilities (squared magnitudes) given real and imaginary parts of complex numbers
 * @param res Real parts array
 * @param ims Imaginary parts array
 * @param probs Output probabilities array
 */
__kernel
void
_CalculateProbabilities(__global const double *res, __global const double *ims, __global double *probs) {
    const size_t i = get_global_id(0);
    
    probs[i] = CalculateProbability(res[i], ims[i]);
}
