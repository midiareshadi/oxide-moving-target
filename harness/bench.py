#!/usr/bin/env python3
"""
harness/bench.py  -  runtime benchmark for the cuda-oxide vs clang vs nvcc study.

One launcher, three cubins. For each kernel we rebuild a cubin from the COMMITTED
PTX using the same ptxas (-arch=sm_89 -O3) that produced the static SASS, then load
and launch each toolchain's kernel with identical data and launch config. Only the
compiled code differs -- the runtime analogue of the "one shared ptxas" control.

Timing: CUDA events, warmup + several rounds, report median per-launch time and the
effective DRAM bandwidth. Correctness is checked against a host reference.

No admin rights needed for timing. For the deeper profile, wrap with ncu, e.g.:
    sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$(which ncu)" \
        --set basic --launch-count 1 \
        python harness/bench.py --kernel reduction --toolchain oxide --profile

Requires: pip install cuda-python numpy   (ptxas must be on PATH)
"""

import argparse
import csv
import datetime as dt
import os
import statistics
import subprocess
import sys
import tempfile

import numpy as np

try:
    from cuda import cuda
except ImportError:  # newer cuda-python layout
    from cuda.bindings import driver as cuda


# ---------------------------------------------------------------- config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(REPO_ROOT, "results", "artifacts")

ARCH = "sm_89"
PEAK_BW_GBPS = 300.0  # L4 approx peak DRAM bandwidth; for the % column only

N = 1 << 24           # 16,777,216 elements; > L2 so vecadd/gather are DRAM-bound
BLOCK = 256
GRID = (N + BLOCK - 1) // BLOCK
NPART = GRID          # reduction writes one partial per block

TOOLCHAINS = ("oxide", "rustcuda", "clang", "nvcc")
ENTRY = {"vecadd": "vecadd", "gather": "gather", "reduction": "reduce"}

# bytes moved through DRAM (for the effective-bandwidth column)
BYTES = {
    "vecadd": 3 * N * 4,    # read a, b ; write c
    "gather": 3 * N * 4,    # read idx, scattered src ; write out  (approx)
    "reduction": N * 4,     # read input ; partials write is negligible
}


# ---------------------------------------------------------------- cuda helpers

def ck(ret):
    """Unwrap a cuda-python (err, *vals) tuple, raising on error."""
    err = ret[0]
    if int(err) != 0:
        try:
            _, s = cuda.cuGetErrorString(err)
            msg = s.decode() if isinstance(s, bytes) else str(s)
        except Exception:
            msg = str(err)
        raise RuntimeError(f"CUDA error {int(err)}: {msg}")
    rest = ret[1:]
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def build_cubin(kernel, tag, outdir):
    """Rebuild a cubin from the committed PTX with our ptxas. Returns path."""
    ptx = os.path.join(ARTIFACTS, kernel, f"{tag}.ptx")
    if not os.path.exists(ptx):
        raise FileNotFoundError(
            f"missing {ptx} -- on the laptop run `git pull`; on the Studio it should exist"
        )
    cubin = os.path.join(outdir, f"{kernel}.{tag}.cubin")
    subprocess.run(
        ["ptxas", f"-arch={ARCH}", "-O3", ptx, "-o", cubin],
        check=True,
    )
    return ptx, cubin


def ptx_param_count(ptx_path, entry):
    """Count .param entries in the kernel signature -- a guard on the ABI guess."""
    txt = open(ptx_path).read()
    i = txt.find(".entry " + entry)
    if i < 0:
        i = txt.find(entry + "(")
    p = txt.find("(", i)
    q = txt.find(")", p)
    return txt[p:q].count(".param")


def pack(arg_list):
    """arg_list: [(value, np.dtype), ...] -> (addr_of_ptr_array, holders)."""
    holders = [np.array([v], dtype=ty) for (v, ty) in arg_list]
    ptrs = np.array([h.ctypes.data for h in holders], dtype=np.uint64)
    return ptrs, holders  # keep holders alive until after the launch


# ---------------------------------------------------------------- env stamp + CSV

def _exe_version(cmd):
    """First line of `<cmd> --version`, for the env stamp; '<missing>' if absent."""
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True)
        out = (cp.stdout + cp.stderr).strip().splitlines()
        return out[0] if out else "<unknown>"
    except FileNotFoundError:
        return "<missing>"


def env_stamp(dev):
    """One stamp shared by every row. Runtime is hardware/clock-dependent, so the
    CSV is only interpretable with this attached -- same discipline as the static
    CSV's tool-version columns."""
    # device name
    try:
        name = ck(cuda.cuDeviceGetName(256, dev))
        gpu = name.decode(errors="replace").strip("\x00").strip() if isinstance(
            name, bytes) else str(name)
    except Exception:
        gpu = "<unknown>"
    # CUDA driver API version (e.g. 13000 -> 13.0)
    try:
        v = int(ck(cuda.cuDriverGetVersion()))
        cuda_driver = f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:
        cuda_driver = "<unknown>"
    # NVIDIA kernel-driver version, best effort via nvidia-smi
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True)
        nvidia_driver = cp.stdout.strip().splitlines()[0].strip() or "<unknown>"
    except Exception:
        nvidia_driver = "<unknown>"
    # repo commit, best effort
    try:
        cp = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)
        commit = cp.stdout.strip() or "<unknown>"
    except Exception:
        commit = "<unknown>"
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "cuda_driver": cuda_driver,
        "nvidia_driver": nvidia_driver,
        "ptxas": _exe_version(["ptxas", "--version"]),
        "arch": ARCH,
        "n_elems": N,
        "block": BLOCK,
        "grid": GRID,
        "repo_commit": commit,
    }


# column order: identity, then the measurement (spread, not one number), then stamp
CSV_FIELDS = [
    "kernel", "toolchain", "median_us", "min_us", "rounds", "reps",
    "gbps", "pct_peak", "check",
    "n_elems", "block", "grid", "arch",
    "gpu", "cuda_driver", "nvidia_driver", "ptxas", "repo_commit", "timestamp",
]


def write_runtime_csv(rows, path):
    """Read-merge-rewrite keyed by (kernel, toolchain) -- re-runs replace, not
    append -- mirroring build_all.py's static_v1.csv discipline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                merged[(r.get("kernel"), r.get("toolchain"))] = r
    for r in rows:
        merged[(r["kernel"], r["toolchain"])] = r
    ordered = sorted(merged.values(),
                     key=lambda r: (r.get("kernel", ""), r.get("toolchain", "")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow(r)
    print(f"\nwrote {len(rows)} row(s) -> {os.path.relpath(path, REPO_ROOT)}")


# ---------------------------------------------------------------- arg layouts

def args_for(kernel, tag, ptrs, n, npart):
    """Return the [(value, dtype)] list for this kernel + toolchain ABI."""
    u64, u32 = np.uint64, np.uint32
    if kernel == "vecadd":
        a, b, c = ptrs["a"], ptrs["b"], ptrs["c"]
        if tag == "oxide":
            return [(a, u64), (n, u64), (b, u64), (n, u64), (c, u64), (n, u64)]
        if tag == "rustcuda":
            return [(a, u64), (n, u64), (b, u64), (n, u64), (c, u64)]
        return [(a, u64), (b, u64), (c, u64), (n, u32)]
    if kernel == "gather":
        s, idx, out = ptrs["src"], ptrs["idx"], ptrs["out"]
        if tag == "oxide":
            return [(s, u64), (n, u64), (idx, u64), (n, u64), (out, u64), (n, u64)]
        if tag == "rustcuda":
            return [(s, u64), (n, u64), (idx, u64), (n, u64), (out, u64), (n, u64)]
        return [(s, u64), (idx, u64), (out, u64), (n, u32)]
    if kernel == "reduction":
        inp, part = ptrs["in"], ptrs["part"]
        if tag == "oxide":
            return [(inp, u64), (n, u64), (part, u64), (npart, u64)]
        if tag == "rustcuda":
            return [(inp, u64), (n, u64), (part, u64)]
        return [(inp, u64), (part, u64), (n, u32)]
    raise ValueError(kernel)


# ---------------------------------------------------------------- host data

def make_inputs(kernel):
    """Return (host_inputs dict, reference, output_key, output_nbytes)."""
    if kernel == "vecadd":
        a = np.ones(N, np.float32)
        b = np.full(N, 2.0, np.float32)
        ref = a + b  # all 3.0
        return {"a": a, "b": b}, ref, "c", N * 4
    if kernel == "gather":
        src = np.arange(N, dtype=np.float32)          # exact: N <= 2^24
        idx = ((np.arange(N, dtype=np.uint64) * 7919) % N).astype(np.uint32)
        ref = src[idx]
        return {"src": src, "idx": idx}, ref, "out", N * 4
    if kernel == "reduction":
        inp = np.ones(N, np.float32)
        ref = None  # checked as: sum(partials) == N
        return {"in": inp}, ref, "part", NPART * 4
    raise ValueError(kernel)


# ---------------------------------------------------------------- main bench

def bench_kernel(kernel, tags, reps, rounds, warmup, profile, stamp=None):
    entry = ENTRY[kernel]
    host, ref, out_key, out_nbytes = make_inputs(kernel)
    rows = []  # one CSV row per timed toolchain

    # device buffers (allocated once, shared across toolchains)
    dptr = {}
    for name, arr in host.items():
        d = ck(cuda.cuMemAlloc(arr.nbytes))
        ck(cuda.cuMemcpyHtoD(d, arr.ctypes.data, arr.nbytes))
        dptr[name] = d
    d_out = ck(cuda.cuMemAlloc(out_nbytes))
    # name the output buffer per kernel
    out_name = {"vecadd": "c", "gather": "out", "reduction": "part"}[kernel]
    dptr[out_name] = d_out

    stream = ck(cuda.cuStreamCreate(0))
    start = ck(cuda.cuEventCreate(0))
    stop = ck(cuda.cuEventCreate(0))

    ptr_ints = {k: int(v) for k, v in dptr.items()}

    print(f"\n=== {kernel}  (N={N:,}  grid={GRID}  block={BLOCK}) ===")
    print(f"{'toolchain':<10}{'us/launch':>12}{'GB/s':>10}{'% peak':>9}  check")

    tmp = tempfile.mkdtemp(prefix="bench_cubin_")
    for tag in tags:
        ptx, cubin = build_cubin(kernel, tag, tmp)

        # guard: does the PTX signature match the ABI we are about to pack?
        arg_list = args_for(kernel, tag, ptr_ints, N, NPART)
        want = ptx_param_count(ptx, entry)
        if want != len(arg_list):
            raise RuntimeError(
                f"{kernel}/{tag}: PTX has {want} params but we packed {len(arg_list)} "
                f"-- ABI mismatch, refusing to launch"
            )

        mod = ck(cuda.cuModuleLoad(cubin.encode()))
        func = ck(cuda.cuModuleGetFunction(mod, entry.encode()))

        smem = 0  # kernels use static shared memory baked into the cubin

        def launch():
            ptrs, _hold = pack(arg_list)
            ck(cuda.cuLaunchKernel(
                func, GRID, 1, 1, BLOCK, 1, 1, smem, stream,
                ptrs.ctypes.data, 0))

        if profile:
            launch()
            ck(cuda.cuStreamSynchronize(stream))
            print(f"{tag:<10}{'(profiled)':>12}")
            ck(cuda.cuModuleUnload(mod))
            continue

        for _ in range(warmup):
            launch()
        ck(cuda.cuStreamSynchronize(stream))

        per_launch_us = []
        for _ in range(rounds):
            ck(cuda.cuEventRecord(start, stream))
            for _ in range(reps):
                launch()
            ck(cuda.cuEventRecord(stop, stream))
            ck(cuda.cuEventSynchronize(stop))
            ms = ck(cuda.cuEventElapsedTime(start, stop))
            per_launch_us.append((ms / reps) * 1000.0)

        t_us = statistics.median(per_launch_us)
        min_us = min(per_launch_us)
        gbps = BYTES[kernel] / (t_us * 1e-6) / 1e9
        pct = 100.0 * gbps / PEAK_BW_GBPS

        # correctness
        out_host = np.empty(out_nbytes // 4, np.float32)
        ck(cuda.cuMemcpyDtoH(out_host.ctypes.data, d_out, out_nbytes))
        if kernel == "reduction":
            ok = abs(float(out_host.sum()) - float(N)) < 1.0
        else:
            ok = np.array_equal(out_host, ref)
        check = "PASS" if ok else "FAIL"

        print(f"{tag:<10}{t_us:>12.2f}{gbps:>10.1f}{pct:>8.0f}%  {check}")

        row = {
            "kernel": kernel, "toolchain": tag,
            "median_us": round(t_us, 2), "min_us": round(min_us, 2),
            "rounds": rounds, "reps": reps,
            "gbps": round(gbps, 1), "pct_peak": round(pct, 1), "check": check,
        }
        if stamp:
            row.update(stamp)
        rows.append(row)
        ck(cuda.cuModuleUnload(mod))

    # cleanup
    for d in dptr.values():
        ck(cuda.cuMemFree(d))
    ck(cuda.cuStreamDestroy(stream))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="all",
                    choices=["all", "vecadd", "gather", "reduction"])
    ap.add_argument("--toolchain", default="all",
                    choices=["all", "oxide", "rustcuda", "clang", "nvcc"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--profile", action="store_true",
                    help="single launch per toolchain, for wrapping with ncu")
    ap.add_argument("--output", default=os.path.join(
                        REPO_ROOT, "results", "raw", "runtime_v1.csv"),
                    help="CSV path for the timing rows (default results/raw/runtime_v1.csv)")
    ap.add_argument("--no-csv", action="store_true",
                    help="print the table but do not write the CSV")
    args = ap.parse_args()

    kernels = (["vecadd", "gather", "reduction"]
               if args.kernel == "all" else [args.kernel])
    tags = list(TOOLCHAINS) if args.toolchain == "all" else [args.toolchain]

    ck(cuda.cuInit(0))
    dev = ck(cuda.cuDeviceGet(0))
    ctx = ck(cuda.cuDevicePrimaryCtxRetain(dev))  # stable across cuda-python versions
    ck(cuda.cuCtxSetCurrent(ctx))
    # one stamp for the whole run; runtime is hardware/clock-dependent so every row carries it
    stamp = None if (args.profile or args.no_csv) else env_stamp(dev)
    all_rows = []
    try:
        for k in kernels:
            all_rows += bench_kernel(
                k, tags, args.reps, args.rounds, args.warmup, args.profile, stamp)
    finally:
        cuda.cuDevicePrimaryCtxRelease(dev)

    if all_rows and not args.no_csv and not args.profile:
        write_runtime_csv(all_rows, args.output)
    print()


if __name__ == "__main__":
    main()
