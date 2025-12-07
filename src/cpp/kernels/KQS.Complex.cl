inline
double
CalculateProbability(const double re, const double im) {
    return re * re + im * im;
}

__kernel
void
_CalculateProbabilities(__global const double *res, __global const double *ims, __global double *probs) {
    const int i = get_global_id(0);
    
    probs[i] = CalculateProbability(res[i], ims[i]);
}
