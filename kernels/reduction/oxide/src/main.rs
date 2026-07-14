// reduction — cuda-oxide side of the three-way comparison.
//
// Block-wise sum reduction via the classic shared-memory tree. Each block
// reduces BLOCK elements of `input` into one partial sum, written to
// `partials[blockIdx]`; the host sums the partials. This is the first kernel
// that exercises shared memory (addrspace 3), a barrier (bar.sync), and a
// loop, so it tests whether cuda-oxide's fixed prologue amortizes once there
// is real per-thread work.
//
// Build-time watch-points (could not be compiled offline):
//   * SMEM size (256) MUST equal BLOCK; const generics want a literal here.
//   * sync_threads() sits OUTSIDE the `if tid < s` — a divergent barrier
//     deadlocks.
//   * shared-memory access is `unsafe` (static mut + uninitialised + races).
//   * the output write uses get_unchecked_mut: the slot is the *block* index,
//     not this thread's ThreadIndex, so the safe get_mut(ThreadIndex) path
//     does not apply.

use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

const BLOCK: usize = 256; // must match the SharedArray size below

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn reduce(input: &[f32], mut partials: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = (thread::blockIdx_x() * thread::blockDim_x()
            + thread::threadIdx_x()) as usize;

        // Stage 1 — each thread loads one element (0.0 past the end) into shared.
        let v = if gid < input.len() { input[gid] } else { 0.0 };
        unsafe { SMEM[tid] = v; }
        thread::sync_threads();

        // Stage 2 — in-place tree reduction. The barrier is OUTSIDE the `if`
        // so every thread in the block reaches it.
        let mut s = (thread::blockDim_x() / 2) as usize;
        while s > 0 {
            if tid < s {
                unsafe { SMEM[tid] = SMEM[tid] + SMEM[tid + s]; }
            }
            thread::sync_threads();
            s >>= 1;
        }

        // Stage 3 — thread 0 writes this block's partial sum to partials[blockIdx].
        // Block-indexed, single-writer: not the per-thread ThreadIndex pattern,
        // so this is the unchecked write path.
        if tid == 0 {
            let bid = thread::blockIdx_x() as usize;
            unsafe { *partials.get_unchecked_mut(bid) = SMEM[0]; }
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();

    const N: usize = 1 << 20; // 1,048,576 elements (an exact multiple of BLOCK)
    let blocks = (N + BLOCK - 1) / BLOCK;

    // Inputs in 0..7 so every per-block partial sum is a small exact integer
    // in f32 — the correctness check is then a clean equality, not a tolerance.
    let host_in: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let expected: f64 = host_in.iter().map(|&x| x as f64).sum();

    let in_dev = DeviceBuffer::from_host(&stream, &host_in).unwrap();
    let mut partials_dev = DeviceBuffer::<f32>::zeroed(&stream, blocks).unwrap();

    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0, // static SharedArray, not dynamic
    };
    module
        .reduce(&stream, cfg, &in_dev, &mut partials_dev)
        .expect("launch");

    let partials = partials_dev.to_host_vec(&stream).unwrap();
    let got: f64 = partials.iter().map(|&x| x as f64).sum();

    if (got - expected).abs() <= 1e-3 * expected.abs().max(1.0) {
        println!("PASSED: reduce sum = {got} (expected {expected})");
    } else {
        println!("FAILED: reduce sum = {got}, expected {expected}");
        std::process::exit(1);
    }
}
