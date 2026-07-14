// vecadd — cuda-oxide (Tier 1: race-free by construction).
//
// This is the Rust side of the three-way PTX comparison. The kernel is
// deliberately the simplest possible elementwise op so it acts as the
// calibration baseline: all three toolchains should produce essentially
// the same code here, and `ptxas -v` should report the same register
// count. Any later kernel where they diverge is then meaningful.
//
// API verified against the cuda-oxide docs (launching-kernels / hello-gpu).
// If the project's API has moved since, regenerate the skeleton with
// `cargo oxide new vecadd` and drop this kernel body back in.

use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[cuda_module]
mod kernels {
    use super::*;

    // Tier 1: the DisjointSlice<f32> + ThreadIndex pair makes the write
    // provably race-free without `unsafe`. The `get_mut(idx)` returns an
    // Option, so the bounds check lives in the type system. The whole
    // point of the study is to see whether that Option check survives to
    // PTX/SASS or whether LLVM elides it (the "is safety zero-cost?" test).
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

fn main() {
    const N: usize = 1024;

    let ctx = CudaContext::new(0).expect("no CUDA device (need CC 8.0+, e.g. L4)");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load embedded PTX module");

    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    module
        .vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a, &b, &mut c)
        .expect("kernel launch failed");

    let c_host = c.to_host_vec(&stream).unwrap();
    let errors = (0..N).filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5).count();

    if errors == 0 {
        println!("\u{2713} SUCCESS: all {N} elements correct");
    } else {
        eprintln!("\u{2717} FAIL: {errors} mismatched elements");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Phase 2 (safety axis): add a Tier-3 raw-pointer variant of the same kernel
// as a second bin target, e.g. src/bin/vecadd_raw.rs, and diff its PTX/SASS
// against this one. Same math, `unsafe` indexing, no DisjointSlice. The delta
// (if any) is the measured cost of the safe abstraction.
// ---------------------------------------------------------------------------
