use cuda_std::prelude::*;
use cuda_std::address_space;
use core::mem::MaybeUninit;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn reduce(input: &[f32], partials: *mut f32) {
    #[address_space(shared)]
    static mut SMEM: [MaybeUninit<f32>; 256] = [const { MaybeUninit::uninit() }; 256];

    let smem = (&raw mut SMEM) as *mut f32;

    let tid = thread::thread_idx_x() as usize;
    let gid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;

    let v = if gid < input.len() { input[gid] } else { 0.0 };
    *smem.add(tid) = v;
    thread::sync_threads();

    let mut s = (thread::block_dim_x() / 2) as usize;
    while s > 0 {
        if tid < s {
            *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s);
        }
        thread::sync_threads();
        s >>= 1;
    }

    if tid == 0 {
        let bid = thread::block_idx_x() as usize;
        *partials.add(bid) = *smem.add(0);
    }
}
