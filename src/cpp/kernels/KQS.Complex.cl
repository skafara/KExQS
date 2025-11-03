__kernel void _CalculateProbabilities(__global const double *res, __global const double *ims, __global double *probs) {
    const int i = get_global_id(0);
    
    probs[i] = res[i] * res[i] + ims[i] * ims[i];
}
