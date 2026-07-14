// vecadd — CUDA C++ side of the three-way comparison.
//
// Compiled twice: once by clang (--cuda-device-only, upstream LLVM/NVPTX)
// and once by nvcc (cicc + libNVVM). The harness emits PTX from both, then
// runs the SAME ptxas over all three PTX files (these two plus cuda-oxide's)
// so register/SASS differences reflect the PTX, not the ptxas version.
//
// Kept semantically identical to the Rust kernel: one thread per element,
// a single in-bounds guard. The `if (i < n)` guard is the C++ analogue of
// the DisjointSlice get_mut Option check on the Rust side.

extern "C" __global__ void vecadd(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Phase 1 refinement: to match register allocation across toolchains, add a
// launch-bounds hint that mirrors the block size in configs/vecadd.yaml, e.g.
//   __global__ void __launch_bounds__(256) vecadd(...)
// ptxas reads the resulting .maxntid directive from every PTX file uniformly.
