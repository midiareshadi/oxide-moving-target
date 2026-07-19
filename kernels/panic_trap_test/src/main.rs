// panic_trap_test: src.get(j).unwrap() with j out of range -> None -> must panic/trap.
// Broken backend: opt -O2 deletes the panic path -> launch "succeeds", out[0] = garbage.
// Fixed backend (#408): the kernel traps. Repro for issue #407.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn unwrap_get(src: &[f32], j: usize, mut out: DisjointSlice<f32>) {
        if let Some(out_elem) = out.get_mut(thread::index_1d()) {
            *out_elem = src.get(j).copied().unwrap(); // None when j >= len -> must panic
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let src = DeviceBuffer::from_host(&stream, &[1.0f32, 2.0, 3.0, 4.0]).expect("src");
    let mut out = DeviceBuffer::from_host(&stream, &[-1.0f32]).expect("out");
    let result = unsafe {
        module.unwrap_get(&stream, LaunchConfig::for_num_elems(1),
                          &src, 1_000_000usize, &mut out)
    }.and_then(|()| out.to_host_vec(&stream));
    match result {
        Err(e) => println!("PASS (kernel trapped: {})", e),
        Ok(out) => {
            println!("FAIL (unwrap(None) did not panic, out[0] = {})", out[0]);
            std::process::exit(1);
        }
    }
}
