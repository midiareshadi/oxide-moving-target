use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let i = thread::index_1d() as usize;
    if i < a.len() {
        *c.add(i) = a[i] + b[i];
    }
}
