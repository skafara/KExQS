__kernel void _SampleAliasTable(__global const double *probs, __global const uint *aliases, __global const uint *bins, __global const double *rands, __global uint *samples) {
    const int i = get_global_id(0);
    
    samples[i] = (rands[i] < probs[bins[i]]) ? bins[i] : aliases[bins[i]];
}
