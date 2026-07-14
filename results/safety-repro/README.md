# Safety repro: gather out-of-bounds read (July backend)

Backend: 29396b7 (heads/main, 2026-07-04), frontend cargo-oxide v0.2.1.
GPU: NVIDIA L4, driver 580.159.03, arch sm_89.

## What this shows
The gather kernel uses SAFE Rust checked indexing: `src[indices[i] as usize]`.
The index is loaded from device memory, so it is not statically provable in
range. In June, cuda-oxide emitted bounds-check traps for this access. On the
July backend, those traps are gone (gather PTX: exit/trap = 0).

## The test
Copy of the gather kernel with one change: indices[0] = 1_000_000, far past
src.len() = 1024. Then run on the GPU.

## Result
- Plain run: DriverError(700, "an illegal memory access was encountered").
- compute-sanitizer memcheck:
    "Invalid __global__ read of size 4 bytes at gather+0x110 ...
     Access is out of bounds ... 3987713 bytes after the nearest allocation
     of size 4096 bytes."
  (1024 floats = 4096 bytes; index 1_000_000 * 4 = ~4 MB past the start.)

Safe Rust indexing performed an out-of-bounds global read. In June this
access trapped; on the July backend it does not.

See compute-sanitizer-gather-oob.txt for the full tool output.

## Contrast: June v0.2.0 backend (faea395, 2026-06-05), SAME test

- compute-sanitizer memcheck: ERROR SUMMARY: 0 errors.
- Program output: "Kernel finished without trapping."
    out[0] (from OOB index 1_000_000) = 0   <- safely guarded, default value
    out[1] (valid) = 1022                    <- correct
- gather PTX on this backend: exit/trap = 3, generic addressing.

So on v0.2.0 the out-of-bounds access is safely guarded (zero sanitizer
errors); on the July main backend the identical program performs an invalid
out-of-bounds global read. Same safe-Rust source, opposite memory safety.

Bracket: v0.2.0 (June 5) is SAFE; a9748f4 (June 11) already emits the
no-check code. The regression landed in the first six days after v0.2.0.
