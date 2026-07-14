use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn gather(src: &[f32], indices: &[u32], out: *mut f32, n: usize) {
    let i = thread::index_1d() as usize;
    if i < n {
        let j = indices[i] as usize;
        *out.add(i) = src[j];
    }
}
