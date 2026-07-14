use std::env;
use std::path::PathBuf;

use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    CudaBuilder::new(manifest_dir.join("kernels"))
        .arch(NvvmArch::Compute89)
        .copy_to(manifest_dir.join("reduction.ptx"))
        .build()
        .unwrap();
}
