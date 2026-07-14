// reduction — CUDA C++ side of the three-way comparison.
//
// A faithful translation of the cuda-oxide kernel: the classic shared-memory
// tree reduction. Compiled by clang (--cuda-device-only, upstream LLVM/NVPTX)
// and by nvcc (cicc + libNVVM); the harness emits PTX from both and runs the
// SAME ptxas over all three PTX files.
//
// Kept semantically identical to the Rust version:
//   * static __shared__ float[256]  ==  static mut SharedArray<f32, 256>
//   * (gid < n) ? input[gid] : 0.0f  ==  the padded load
//   * __syncthreads() OUTSIDE the `if (tid < s)`  ==  non-divergent barrier
//   * partials[blockIdx.x] = smem[0] ==  the block-indexed, thread-0 write
//
// The one faithful difference is `int n`: C++ raw pointers do not carry a
// length, so the bound is passed explicitly, where the Rust slice supplies
// input.len(). That asymmetry is exactly what the study isolates.
extern "C" __global__ void reduce(const float* input,
                                  float* partials,
                                  int n) {
    __shared__ float smem[256];

    unsigned tid = threadIdx.x;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < (unsigned)n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = smem[0];
    }
}
