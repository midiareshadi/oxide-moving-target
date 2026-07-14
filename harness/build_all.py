#!/usr/bin/env python3
"""
build_all.py — three-way PTX/SASS harness for the cuda-oxide study.

For one kernel (driven by a configs/<kernel>.yaml file) this:
  1. builds the cuda-oxide Rust kernel  -> oxide.ptx (+ oxide.ll if available)
  2. compiles the CUDA C++ with clang   -> clang.ptx + clang.ll
  3. compiles the CUDA C++ with nvcc     -> nvcc.ptx
  4. runs the SAME ptxas over all three PTX files -> cubin + ptxas -v report
  5. disassembles each cubin              -> SASS
  6. parses static metrics via analysis/diff_metrics.py
  7. writes artifacts to results/artifacts/<kernel>/ and upserts a row in
     results/raw/static_v1.csv, stamping every tool version on the row.

The cuda-oxide vs clang pair shares the upstream LLVM backend, so it isolates
the language. clang vs nvcc isolates the backend. nvcc's internal NVVM .ll is
not readily dumpable, so the .ll-level diff is cuda-oxide vs clang only.

Nothing here is GPU-bound except the optional --run check, so the whole thing
works on a free CPU Studio; switch to the L4 only for that step.

Usage:
    python3 -m harness.build_all --config configs/vecadd.yaml
    python3 -m harness.build_all --config configs/vecadd.yaml --no-run
    python3 -m harness.build_all --from-artifacts   # rebuild CSV, no toolchain
"""

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# canonical CSV schema — fixed column layout so the file is always
# self-describing and every row shares one schema regardless of which
# toolchains ran. Metric-key order mirrors diff_metrics.PTX_CLASSES.
# --------------------------------------------------------------------------- #
META_COLS = ["kernel", "timestamp", "arch", "opt", "fast_math",
             "repo_commit", "llc", "clang", "nvcc", "ptxas"]
TAGS = ("oxide", "rustcuda", "clang", "nvcc")

_PTX_KEYS = [
    "ptx_ld_param", "ptx_st_param", "ptx_ld_global", "ptx_st_global",
    "ptx_ld_shared", "ptx_st_shared", "ptx_ld_generic", "ptx_st_generic",
    "ptx_fma", "ptx_mul", "ptx_add", "ptx_branch", "ptx_barrier",
    "ptx_atomic", "ptx_setp", "ptx_sfu", "ptx_mov", "ptx_cvt",
    "ptx_instr_total", "ptx_labels", "ptx_panic_refs", "ptx_trap_refs",
]
_LL_KEYS = ["ll_lines", "ll_basic_blocks", "ll_panic_refs",
            "ll_trap_refs", "ll_landingpads"]
_PTXAS_KEYS = ["ptxas_regs", "ptxas_spill_stores", "ptxas_spill_loads",
               "ptxas_smem", "ptxas_cmem", "ptxas_stack"]
_SASS_KEYS = ["sass_instr_total"]
_CANON_METRICS = _PTX_KEYS + _LL_KEYS + _PTXAS_KEYS + _SASS_KEYS


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def run(cmd, **kw):
    """Run a command, echoing it. Returns CompletedProcess (never raises)."""
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def tool_version(cmd):
    """Best-effort version string for env stamping; '<missing>' if absent."""
    exe = cmd[0]
    if shutil.which(exe) is None:
        return "<missing>"
    cp = subprocess.run(cmd, capture_output=True, text=True)
    out = (cp.stdout + cp.stderr).strip().splitlines()
    return out[0] if out else "<unknown>"


def git_commit(path):
    cp = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True)
    return cp.stdout.strip() or "<unknown>"


def need(*exes):
    """Warn (don't die) about missing tools so partial runs still work."""
    missing = [e for e in exes if shutil.which(e) is None]
    if missing:
        print(f"  ! not found: {', '.join(missing)} — skipping that toolchain")
    return not missing


# --------------------------------------------------------------------------- #
# per-toolchain builds. Each returns dict of {artifact_name: Path or None}.
# --------------------------------------------------------------------------- #
def build_oxide(cfg, out):
    """cuda-oxide: Rust -> PTX + LLVM IR, written to the project dir as <pkg>.ptx/.ll."""
    print("[cuda-oxide]")
    if not need("cargo"):
        return {}
    rust_dir = REPO / cfg["rust_dir"]
    kernel = cfg["kernel"]
    run(["cargo", "oxide", "build", "--arch", cfg["arch"]], cwd=rust_dir)
    oxide_ptx, oxide_ll = out / "oxide.ptx", out / "oxide.ll"
    src_ptx = rust_dir / f"{kernel}.ptx"
    if not src_ptx.exists():
        c = sorted(rust_dir.glob("*.ptx")); src_ptx = c[-1] if c else None
    if src_ptx:
        shutil.copy(src_ptx, oxide_ptx)
    else:
        print(f"  ! no PTX produced in {rust_dir}"); oxide_ptx = None
    src_ll = rust_dir / f"{kernel}.ll"
    if not src_ll.exists():
        c = sorted(rust_dir.glob("*.ll")); src_ll = c[-1] if c else None
    if src_ll:
        shutil.copy(src_ll, oxide_ll)
    else:
        print(f"  ! no LLVM IR produced in {rust_dir}"); oxide_ll = None
    return {"oxide.ptx": oxide_ptx, "oxide.ll": oxide_ll}


def build_rustcuda(cfg, out):
    """rust-cuda: Rust -> PTX via libNVVM. Built OUT-OF-BAND (needs the LLVM-7
    environment in ~/rust-cuda-env.sh and is slow), so here we only COLLECT the
    already-built PTX. Build it once with:
        source ~/rust-cuda-env.sh
        (cd kernels/<k>/rust-cuda && cargo build)
    No comparable upstream .ll (libNVVM path), so PTX only — like nvcc."""
    print("[rust-cuda]  (collect pre-built PTX)")
    kernel = cfg["kernel"]
    rc_dir = REPO / "kernels" / kernel / "rust-cuda"
    src_ptx = rc_dir / f"{kernel}.ptx"
    rustcuda_ptx = out / "rustcuda.ptx"
    if src_ptx.exists():
        shutil.copy(src_ptx, rustcuda_ptx)
        return {"rustcuda.ptx": rustcuda_ptx}
    print(f"  ! no pre-built PTX at {src_ptx.relative_to(REPO)} "
          f"— run: source ~/rust-cuda-env.sh && "
          f"(cd kernels/{kernel}/rust-cuda && cargo build)")
    return {"rustcuda.ptx": None}


def build_clang(cfg, out):
    """clang CUDA: C++ -> PTX and LLVM IR, upstream NVPTX backend."""
    print("[clang]")
    clang = "clang++"
    if not need(clang):
        return {}
    src = REPO / cfg["cuda_src"]
    arch = cfg["arch"]
    opt = f"-O{cfg.get('opt', 3)}"
    # This Studio's conda clang refuses to auto-detect the split conda CUDA
    # layout as a CUDA root, so we drive it fully explicitly: skip clang's
    # CUDA include/lib detection (-nocudainc -nocudalib), supply the toolkit
    # headers, force-include clang's CUDA wrapper (defines __global__ etc.),
    # and link libdevice directly. Paths come from the env (CUDA_TOOLKIT_INCLUDE,
    # CLANG_CUDA_WRAPPER, LIBDEVICE_BC) with conda defaults.
    conda = os.environ.get("CONDA_PREFIX", "")
    inc = os.environ.get("CUDA_TOOLKIT_INCLUDE",
                         f"{conda}/targets/x86_64-linux/include")
    libdevice = os.environ.get("LIBDEVICE_BC",
                         f"{conda}/nvvm/libdevice/libdevice.10.bc")
    wrapper = os.environ.get("CLANG_CUDA_WRAPPER", "")
    if not wrapper and conda:
        cands = sorted(Path(conda).glob(
            "lib/clang/*/include/__clang_cuda_runtime_wrapper.h"))
        wrapper = str(cands[-1]) if cands else ""
    common = [clang, "-x", "cuda", f"--cuda-gpu-arch={arch}",
              "--cuda-device-only", "-nocudainc", "-nocudalib",
              "-isystem", inc,
              "-Xclang", "-mlink-builtin-bitcode", "-Xclang", libdevice,
              "-Wno-unknown-cuda-version", opt]
    if wrapper:
        common += ["-include", wrapper]
    else:
        print("  ! clang CUDA wrapper header not found; clang cell may fail")
    if cfg.get("fast_math"):
        common.append("-ffast-math")

    clang_ptx = out / "clang.ptx"
    clang_ll = out / "clang.ll"
    run(common + ["-S", str(src), "-o", str(clang_ptx)])          # -S => PTX
    run(common + ["-emit-llvm", "-S", str(src), "-o", str(clang_ll)])
    return {"clang.ptx": clang_ptx if clang_ptx.exists() else None,
            "clang.ll": clang_ll if clang_ll.exists() else None}


def build_nvcc(cfg, out):
    """nvcc: C++ -> PTX via cicc + libNVVM (no readily-dumpable NVVM .ll)."""
    print("[nvcc]")
    if not need("nvcc"):
        return {}
    src = REPO / cfg["cuda_src"]
    arch = cfg["arch"]
    opt = f"-O{cfg.get('opt', 3)}"
    cmd = ["nvcc", f"-arch={arch}", "-ptx", opt]
    if cfg.get("fast_math"):
        cmd.append("--use_fast_math")
    nvcc_ptx = out / "nvcc.ptx"
    run(cmd + [str(src), "-o", str(nvcc_ptx)])
    return {"nvcc.ptx": nvcc_ptx if nvcc_ptx.exists() else None}


# --------------------------------------------------------------------------- #
# shared ptxas pass + SASS — the key control: ONE ptxas over every PTX file
# --------------------------------------------------------------------------- #
def assemble_and_disasm(cfg, out, ptx_files):
    """Run identical ptxas on each PTX -> cubin + reg report, then SASS."""
    print("[ptxas + SASS]  (one ptxas over all PTX -> isolates ptxas noise)")
    arch = cfg["arch"]
    reports = {}
    if not need("ptxas"):
        return reports
    for name, ptx in ptx_files.items():
        if ptx is None or not Path(ptx).exists():
            continue
        tag = name.split(".")[0]                # oxide / clang / nvcc
        cubin = out / f"{tag}.cubin"
        v = run(["ptxas", f"-arch={arch}", "-O3", "-v",
                 str(ptx), "-o", str(cubin)])
        (out / f"{tag}.ptxas.txt").write_text(v.stdout + v.stderr)  # -v -> stderr
        reports[tag] = v.stdout + v.stderr
        if cubin.exists() and shutil.which("cuobjdump"):
            sass = run(["cuobjdump", "-sass", str(cubin)])
            (out / f"{tag}.sass").write_text(sass.stdout)
    return reports


# --------------------------------------------------------------------------- #
# metrics + CSV
# --------------------------------------------------------------------------- #
def collect_metrics(cfg, out):
    """Hand the artifacts to diff_metrics for a per-toolchain metrics dict."""
    try:
        from analysis.diff_metrics import metrics_for_dir
    except Exception as e:                       # noqa: BLE001
        print(f"  ! diff_metrics unavailable ({e}); writing artifacts only")
        return {}
    return metrics_for_dir(out)


def env_stamp(cfg):
    rust_dir = REPO / cfg["rust_dir"]
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "arch": cfg["arch"],
        "opt": cfg.get("opt", 3),
        "fast_math": cfg.get("fast_math", False),
        # the study repo's HEAD at build time (provenance for the artifacts);
        # not the cuda-oxide compiler version, hence 'repo_commit' not 'oxide_*'.
        "repo_commit": git_commit(rust_dir),
        "llc": tool_version(["llc-21", "--version"]),
        "clang": tool_version(["clang++", "--version"]),
        "nvcc": tool_version(["nvcc", "--version"]),
        "ptxas": tool_version(["ptxas", "--version"]),
    }


def _ordered_metric_keys(found):
    """Canonical metric order first; any unrecognised keys appended (sorted)."""
    extra = [k for k in sorted(found) if k not in _CANON_METRICS]
    return _CANON_METRICS + extra


def _csv_path():
    return REPO / "results" / "raw" / "static_v1.csv"


def _load_rows(csv_path):
    """Read existing CSV into {kernel: row-dict}, keyed by kernel (last wins).

    Relies only on named columns, so it tolerates the older malformed file
    (any unnamed trailing values are dropped — metrics are recomputed anyway)."""
    rows = {}
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r.get("kernel") or "").strip()
            if k:
                rows[k] = {kk: vv for kk, vv in r.items()
                           if kk is not None and vv is not None}
    return rows


def _row_from(stamp, metrics, kernel):
    """Flatten a stamp + {tag: metrics} into one wide row dict."""
    row = {"kernel": kernel}
    for col in META_COLS:
        if col != "kernel":
            row[col] = stamp.get(col, "")
    for tag, m in metrics.items():               # tag = oxide/clang/nvcc
        for k, v in m.items():
            row[f"{tag}_{k}"] = v
    return row


def _write_csv(csv_path, rows_by_kernel):
    """Rewrite the whole CSV with a complete canonical header, one row/kernel."""
    found = set()
    for row in rows_by_kernel.values():
        for col in row:
            head, _, tail = col.partition("_")
            if head in TAGS and tail:
                found.add(tail)
    fields = list(META_COLS) + [f"{t}_{k}"
                                for t in TAGS
                                for k in _ordered_metric_keys(found)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for kernel in sorted(rows_by_kernel):
            w.writerow(rows_by_kernel[kernel])


def write_row(cfg, stamp, metrics):
    """Upsert one row for this kernel, then rewrite the CSV cleanly.

    Replaces (does not append) the kernel's row, so re-runs don't duplicate,
    and always writes a complete self-describing header."""
    csv_path = _csv_path()
    rows = _load_rows(csv_path)
    rows[cfg["kernel"]] = _row_from(stamp, metrics, cfg["kernel"])
    _write_csv(csv_path, rows)
    print(f"\nUpserted row for '{cfg['kernel']}' -> {csv_path.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# regeneration from committed artifacts (no toolchain / no GPU)
# --------------------------------------------------------------------------- #
def _arch_from_ptx(art_dir):
    for ptx in sorted(art_dir.glob("*.ptx")):
        m = re.search(r"\.target\s+(sm_\d+)", ptx.read_text(errors="replace"))
        if m:
            return m.group(1)
    return ""


def regenerate_from_artifacts():
    """Rebuild static_v1.csv purely from the committed results/artifacts/*.

    Recomputes every metric from the text artifacts and salvages provenance
    (timestamp / tool versions / commit) from the existing CSV where present."""
    from analysis.diff_metrics import metrics_for_dir
    csv_path = _csv_path()
    salvage = _load_rows(csv_path)               # metadata only
    art_root = REPO / "results" / "artifacts"
    if not art_root.exists():
        sys.exit(f"no artifacts dir at {art_root}")
    rows = {}
    for d in sorted(p for p in art_root.iterdir() if p.is_dir()):
        metrics = metrics_for_dir(d)
        if not metrics:
            continue
        kernel = d.name
        old = salvage.get(kernel, {})
        stamp = {
            "timestamp": old.get("timestamp") or "regenerated-from-artifacts",
            "arch": old.get("arch") or _arch_from_ptx(d),
            "opt": old.get("opt", ""),
            "fast_math": old.get("fast_math", ""),
            # old files used the column name 'oxide_commit' for the same thing
            "repo_commit": old.get("repo_commit") or old.get("oxide_commit", ""),
            "llc": old.get("llc", ""),
            "clang": old.get("clang", ""),
            "nvcc": old.get("nvcc", ""),
            "ptxas": old.get("ptxas", ""),
        }
        rows[kernel] = _row_from(stamp, metrics, kernel)
    _write_csv(csv_path, rows)
    print(f"Regenerated {csv_path.relative_to(REPO)} from {len(rows)} kernel(s): "
          f"{', '.join(sorted(rows))}")


def runtime_check(cfg):
    """Optional: actually launch the cuda-oxide kernel on the GPU (needs L4)."""
    print("[runtime check]  (needs a CC 8.0+ GPU — switch the Studio to L4)")
    rust_dir = REPO / cfg["rust_dir"]
    profile = cfg.get("oxide_profile", "release")
    cmd = ["cargo", "oxide", "run", cfg["kernel"]]
    if profile == "release":
        cmd.append("--release")
    cp = run(cmd, cwd=rust_dir)
    print(cp.stdout.strip() or cp.stderr.strip())
    return cp.returncode == 0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="three-way PTX/SASS harness")
    ap.add_argument("--config", help="path to configs/<kernel>.yaml")
    ap.add_argument("--from-artifacts", action="store_true",
                    help="rebuild static_v1.csv from committed artifacts only "
                         "(no toolchain, no GPU)")
    ap.add_argument("--no-run", action="store_true",
                    help="skip the GPU runtime check (CPU-only studio)")
    args = ap.parse_args()

    if args.from_artifacts:
        regenerate_from_artifacts()
        return

    if not args.config:
        ap.error("--config is required unless --from-artifacts is given")

    try:
        import yaml
    except ImportError:
        sys.exit("Missing pyyaml. Run: pip install -r requirements.txt")

    cfg = yaml.safe_load((REPO / args.config).read_text()
                         if not os.path.isabs(args.config)
                         else Path(args.config).read_text())

    out = REPO / "results" / "artifacts" / cfg["kernel"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {cfg['kernel']}  (arch={cfg['arch']}, -O{cfg.get('opt', 3)}) ===")

    artifacts = {}
    artifacts.update(build_oxide(cfg, out))
    artifacts.update(build_rustcuda(cfg, out))
    artifacts.update(build_clang(cfg, out))
    artifacts.update(build_nvcc(cfg, out))

    ptx_files = {k: v for k, v in artifacts.items() if k.endswith(".ptx")}
    assemble_and_disasm(cfg, out, ptx_files)

    stamp = env_stamp(cfg)
    metrics = collect_metrics(cfg, out)
    write_row(cfg, stamp, metrics)

    if cfg.get("runtime_check", True) and not args.no_run:
        runtime_check(cfg)

    print(f"\nArtifacts in {out.relative_to(REPO)}/  — commit these as the evidence.")


if __name__ == "__main__":
    main()
