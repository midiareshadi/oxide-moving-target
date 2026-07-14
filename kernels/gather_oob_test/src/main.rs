use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[cuda_module]
mod kernels {
    use super::*;
    // Safety-stress kernel: out[i] = src[indices[i]].
    // The index into `src` is data-dependent, so bounds are NOT statically
    // provable -- where safe Rust indexing should cost the most vs C++.
    #[kernel]
    pub fn gather(src: &[f32], indices: &[u32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let j = indices[i] as usize;
            *out_elem = src[j];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    const N: usize = 1024;
    let src_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mut idx_host: Vec<u32> = (0..N).map(|i| (N - 1 - i) as u32).collect();
    // DELIBERATE out-of-bounds index: src has 1024 elements; 1_000_000 is far past the end.
    // In safe Rust, src[j] must bounds-check. June cuda-oxide trapped here; testing July.
    idx_host[0] = 1_000_000;
    let src_dev = DeviceBuffer::from_host(&stream, &src_host).unwrap();
    let idx_dev = DeviceBuffer::from_host(&stream, &idx_host).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .gather(&stream, LaunchConfig::for_num_elems(N as u32),
                &src_dev, &idx_dev, &mut out_dev)
        .expect("Kernel launch failed");
    let out_host = out_dev.to_host_vec(&stream).unwrap();
    // out[0] used index 1_000_000 into a 1024-element array.
    // If bounds-checked: the kernel would have trapped (no clean finish).
    // If NOT checked: we get here, and out[0] holds whatever was at that OOB address.
    println!("Kernel finished without trapping.");
    println!("out[0] (from OOB index 1_000_000 into src[0..1024]) = {}", out_host[0]);
    println!("out[1] (valid, should be {}) = {}", src_host[idx_host[1] as usize], out_host[1]);
}
