# oxide-moving-target

**The finding, in one line:** I re-tested [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (a Rust-to-GPU compiler) a few weeks after my [first study](https://github.com/midiareshadi/rust-cuda-vs-oxide). Two of my three findings had changed — and one change removed a bounds check, so safe Rust code now reads out of bounds. Reported as [cuda-oxide issue #396](https://github.com/NVlabs/cuda-oxide/issues/396).

## The reproducibility trap

cuda-oxide has a **frontend** (`cargo-oxide`, version-tagged) and a **backend** (`librustc_codegen_cuda`, the part that makes GPU code). Installing a tagged frontend does **not** pin the backend: the backend is fetched from the latest `main` branch. I confirmed this — frontend v0.2.0 and v0.2.1 both fetched the **same** backend commit (`29396b7`, 2026-07-04). So the version tag does not identify the compiler; the backend commit + date does.

This repo therefore names builds by backend commit, not by version tag:
- **older build** — `faea395` (released as v0.2.0, 2026-06-05)
- **newer build** — `29396b7` (from `main`, 2026-07-04)

## What changed, across three kernels (older → newer build)

| kernel    | addressing        | PTX size       | bounds check      |
|-----------|-------------------|----------------|-------------------|
| vecadd    | generic → global  | 64 → 37        | — (none)          |
| gather    | generic → global  | 64 → 37        | **2 → 0 (dropped)** |
| reduction | generic → global  | 47 → 55 (setup)| — (none)          |

- **Addressing** improved on all three (generic → global loads/stores, matching clang/nvcc). My earlier claim "cuda-oxide is the only one emitting generic loads" is now out of date.
- **Size** is mixed: vecadd/gather shrank ~42%; reduction grew slightly (one-time loop-setup blocks, not hot-path work).
- **Safety**: the gather kernel dropped its bounds checks. See below.

## The safety regression (issue #396)

The gather kernel is `out[i] = src[indices[i]]`, using **safe, checked** Rust indexing (`src[j]`). `j` is loaded from device memory, so it is not statically provable in range — the check cannot be soundly removed.

I ran the identical program on both builds with an out-of-range index (`indices[0] = 1_000_000` into a 1024-element `src`), under NVIDIA's `compute-sanitizer`:

- **older build (v0.2.0):** `ERROR SUMMARY: 0 errors`. The access is guarded; the out-of-bounds read never happens.
- **newer build (main):** `Invalid __global__ read of size 4 bytes ... 3987713 bytes after the nearest allocation of size 4096`. An out-of-bounds global read.

Same safe-Rust source; opposite memory safety. I bisected the change to within about a week of the v0.2.0 release. Full report: [issue #396](https://github.com/NVlabs/cuda-oxide/issues/396) (also saved here as `ISSUE-396.md`).

Honest scope: the check on the *output* slice (`DisjointSlice::get_mut`, a witness-style access) still works. It is specifically plain `&[T]` checked indexing (`src[j]`) that lost its check.

## Where the evidence lives

```
kernels/vecadd, gather, reduction    the three kernels (oxide + cuda cells)
kernels/gather_oob_test              the safety repro (indices[0] = 1_000_000)
results/artifacts-before-june22/     older-build PTX/SASS (the "before")
results/artifacts-after/             newer-build PTX/SASS (the "after")
results/safety-repro/                compute-sanitizer logs, both builds + a README
ISSUE-396.md                         the filed report
harness/, analysis/, configs/        build + measurement scripts
```

Quick checks:
- Addressing/size: `grep -c "ld.global" results/artifacts-*/gather/oxide.ptx`
- Bounds checks: `grep -c "exit" results/artifacts-before-june22/gather/oxide.ptx` (2) vs `results/artifacts-after/gather/oxide.ptx` (0)

## Reproduce

Build a kernel on the current backend and inspect the PTX:

```
cd kernels/gather/oxide && cargo oxide build --arch sm_89
grep -c "exit" gather.ptx     # bounds-check traps: present in older build, 0 in newer
```

Safety test (needs a GPU):

```
cd kernels/gather_oob_test
compute-sanitizer --tool memcheck cargo oxide run --arch sm_89
```

To pin a specific backend, check out its commit in `~/.cargo/cuda-oxide/src`, `cargo clean`, and rebuild — see `results/safety-repro/README.md` for the exact commits.

## Provenance (record this, not the version tag)

- older build: backend `faea395` (v0.2.0, 2026-06-05)
- newer build: backend `29396b7` (`main`, 2026-07-04); frontend cargo-oxide v0.2.1
- rustc nightly-2026-04-03, LLVM 21.1.8, CUDA 12.8, NVIDIA L4 (driver 580.159.03), `sm_89`

## Origin

The kernels and harness come from [rust-cuda-vs-oxide](https://github.com/midiareshadi/rust-cuda-vs-oxide). This repo adds the July rebuild, the before/after artifacts, and the safety repro.

**Writeup:** https://midiareshadi.github.io/blog/oxide-moving-target/

## License

See [LICENSE](LICENSE).
