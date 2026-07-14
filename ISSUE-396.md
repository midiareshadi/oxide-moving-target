> Filed as NVlabs/cuda-oxide issue #396 on 2026-07-13.
> https://github.com/NVlabs/cuda-oxide/issues/396

# kernel reads out of bounds: checked `&[T]` indexing (`src[j]`) no longer emits a bounds check (regressed since v0.2.0)

## Summary

A `#[kernel]` that indexes a device slice with a data-dependent index using **safe, checked Rust indexing** (`src[j]`) no longer emits a bounds check. On an out-of-range index the kernel performs an **out-of-bounds global read** (CUDA 700) instead of terminating the thread.

I bisected this to the codegen backend (`librustc_codegen_cuda`):
- **v0.2.0 (`faea395`, 2026-06-05):** the bounds check is present; the access is safe.
- **`a9748f4` (2026-06-11):** already gone.
- **current `main` (`29396b7`, 2026-07-04):** still gone.

So the change landed within about a week of the v0.2.0 release.

This looks distinct from #132 (`Option::unwrap_or(&literal)` miscompile) and #392 (a `DisjointSlice::len()` codegen crash): this is plain `&[T]` checked indexing losing its bounds check at runtime.

Suggested labels: `safety`, `miscompile`, `codegen`.

## Kernel

```rust
#[kernel]
pub fn gather(src: &[f32], indices: &[u32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        let j = indices[i] as usize;   // checked index load
        *out_elem = src[j];            // checked index; j is runtime data, not statically provable
    }
}
```

`j` is loaded from device memory, so the compiler cannot prove `j < src.len()`. Under Rust semantics `src[j]` must bounds-check.

## Reproduction

Set one index out of range and run the kernel:

```rust
let mut idx_host: Vec<u32> = (0..1024).map(|i| (1023 - i) as u32).collect();
idx_host[0] = 1_000_000;   // src has 1024 elements
```

```
compute-sanitizer --tool memcheck cargo oxide run --arch sm_89
```

## Observed behavior

**Current `main` (`29396b7`, 2026-07-04):**

```
========= Invalid __global__ read of size 4 bytes
=========     at gather+0x110
=========     by thread (0,0,0) in block (0,0,0)
=========     Access to 0x... is out of bounds
=========     and is 3987713 bytes after the nearest allocation at 0x... of size 4096 bytes
========= ERROR SUMMARY: 6 errors
```

(1024 f32 = 4096 bytes; index 1_000_000 x 4 = ~4 MB past the start — matching the reported offset.)

**v0.2.0 (`faea395`, 2026-06-05), identical program:**

```
Kernel finished without trapping.
out[0] (from OOB index 1_000_000) = 0
========= ERROR SUMMARY: 0 errors
```

The v0.2.0 PTX contains the guard: a `setp.ge.u64` on `indices[i] < src.len()` that branches to a clean `exit` when out of range, so the out-of-bounds load never executes. The current-`main` PTX has no such guard and issues the `ld.global` unconditionally.

## Expected behavior

Out-of-bounds checked indexing should not perform the read. Either the documented panic-to-`trap` behavior, or the v0.2.0 behavior (guard the access and skip it) — not a silent out-of-bounds global read.

## PTX difference (gather kernel)

- v0.2.0: two bounds checks present (`setp.ge.u64` → `exit` for `indices[i] < indices.len()` and for `indices[i] < src.len()`); generic loads.
- current `main`: only the thread guard remains; no bounds checks; `ld.global` on the indexed access.

## Environment

- Backend commits tested: `faea395` (v0.2.0, safe), `a9748f4` (2026-06-11, unsafe), `29396b7` (2026-07-04 `main`, unsafe).
- Frontend: cargo-oxide v0.2.1. rustc nightly-2026-04-03. LLVM 21.1.8. CUDA 12.8. GPU: NVIDIA L4, driver 580.159.03, `sm_89`.

I have the full repro (both kernels, before/after PTX, and both compute-sanitizer logs) and can share the repository link if useful.
