// CUDA C++ gather, the clang/nvcc baseline. src[indices[i]] is an unchecked read.
// Note: cuda-oxide also compiles the safe Rust src[j] to an unchecked device load
// (no bounds check, no panic), so the read paths match on safety. See the writeup.
extern "C" __global__ void gather(const float* src,
                                  const unsigned int* indices,
                                  float* out,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = src[indices[i]];
    }
}
