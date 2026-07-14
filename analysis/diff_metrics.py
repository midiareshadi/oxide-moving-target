#!/usr/bin/env python3
"""
diff_metrics.py — turn the emitted artifacts into comparable static metrics.

Parses the .ptx / .ll / .sass / ptxas-report files in an artifacts dir and
returns one metrics dict per toolchain (oxide / clang / nvcc). These are the
numbers the blog post's tables are built from. Pure text parsing, no GPU.

Can also be run standalone to print a side-by-side table:
    python3 -m analysis.diff_metrics results/artifacts/vecadd
"""

import re
import sys
from pathlib import Path

# PTX opcode classes we bucket instructions into. Order matters: first match
# wins, so more specific patterns (e.g. fma) precede generic ones.
PTX_CLASSES = [
    ("ld_param",   re.compile(r"^\s*ld\.param")),
    ("st_param",   re.compile(r"^\s*st\.param")),
    ("ld_global",  re.compile(r"^\s*ld\.global")),
    ("st_global",  re.compile(r"^\s*st\.global")),
    ("ld_shared",  re.compile(r"^\s*ld\.shared")),
    ("st_shared",  re.compile(r"^\s*st\.shared")),
    ("ld_generic", re.compile(r"^\s*ld\.")),
    ("st_generic", re.compile(r"^\s*st\.")),
    ("fma",        re.compile(r"^\s*(fma|mad)\.")),
    ("mul",        re.compile(r"^\s*mul\.")),
    ("add",        re.compile(r"^\s*(add|sub)\.")),
    ("branch",     re.compile(r"^\s*@?%?\w*\s*bra\b|^\s*bra\b")),
    ("barrier",    re.compile(r"^\s*bar\.")),
    ("atomic",     re.compile(r"^\s*(atom|red)\.")),
    ("setp",       re.compile(r"^\s*setp\.")),
    ("sfu",        re.compile(r"^\s*(ex2|lg2|rcp|rsqrt|sqrt|sin|cos)\.")),
    ("mov",        re.compile(r"^\s*mov\.")),
    ("cvt",        re.compile(r"^\s*cvt\.")),
]

# Heuristics for "did a Rust safety artifact survive into the output?"
PANIC_HINTS = re.compile(r"panic|bounds_check|unwrap|index_out_of|__rust", re.I)
TRAP_HINTS = re.compile(r"\btrap\b|llvm\.trap|\.trap")


def _read(path):
    return path.read_text(errors="replace") if path.exists() else ""


def parse_ptx(text):
    """Instruction histogram + register/branch/panic signals from a PTX file."""
    m = {f"ptx_{name}": 0 for name, _ in PTX_CLASSES}
    m["ptx_instr_total"] = 0
    m["ptx_labels"] = 0          # ~ basic-block count
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if re.match(r"^\$?[A-Za-z_]\w*:\s*$", s):   # a label like $L__BB0_1:
            m["ptx_labels"] += 1
            continue
        if s.startswith((".", "{", "}", "@", "ld.param", "ret", "//")):
            # directives / param loads aren't counted as compute instrs,
            # but a real instruction line ends in ';'
            pass
        if s.endswith(";"):
            m["ptx_instr_total"] += 1
            for name, rx in PTX_CLASSES:
                if rx.match(line):
                    m[f"ptx_{name}"] += 1
                    break
    m["ptx_panic_refs"] = len(PANIC_HINTS.findall(text))
    m["ptx_trap_refs"] = len(TRAP_HINTS.findall(text))
    return m


def parse_ll(text):
    """Lightweight LLVM-IR signals: bb count, panic/landing-pad presence."""
    if not text:
        return {}
    return {
        "ll_lines": len(text.splitlines()),
        "ll_basic_blocks": len(re.findall(r"^\d*:?\s*; <label>|^\w[\w.]*:", text, re.M))
                            or text.count("\nbr ") + 1,
        "ll_panic_refs": len(PANIC_HINTS.findall(text)),
        "ll_trap_refs": len(TRAP_HINTS.findall(text)),
        "ll_landingpads": text.count("landingpad"),
    }


def parse_ptxas(text):
    """Pull register / spill / shared-mem numbers from a `ptxas -v` report."""
    if not text:
        return {}
    out = {}
    for key, rx in [
        ("regs",         r"Used\s+(\d+)\s+registers"),
        ("spill_stores", r"(\d+)\s+bytes\s+spill\s+stores"),
        ("spill_loads",  r"(\d+)\s+bytes\s+spill\s+loads"),
        ("smem",         r"(\d+)\s+bytes\s+smem"),
        ("cmem",         r"(\d+)\s+bytes\s+cmem"),
        ("stack",        r"(\d+)\s+bytes\s+stack\s+frame"),
    ]:
        mobj = re.search(rx, text)
        out[f"ptxas_{key}"] = int(mobj.group(1)) if mobj else 0
    return out


def parse_sass(text):
    if not text:
        return {}
    instrs = [l for l in text.splitlines() if re.search(r"/\*[0-9a-f]{4}\*/", l)]
    return {"sass_instr_total": len(instrs)}


def metrics_for_tag(out_dir, tag):
    """All metrics for one toolchain tag (oxide/clang/nvcc) in a dir."""
    out_dir = Path(out_dir)
    m = {}
    m.update(parse_ptx(_read(out_dir / f"{tag}.ptx")))
    m.update(parse_ll(_read(out_dir / f"{tag}.ll")))
    m.update(parse_ptxas(_read(out_dir / f"{tag}.ptxas.txt")))
    m.update(parse_sass(_read(out_dir / f"{tag}.sass")))
    return m


def metrics_for_dir(out_dir):
    """Returns {tag: metrics_dict} for whichever toolchains produced output."""
    out_dir = Path(out_dir)
    result = {}
    for tag in ("oxide", "rustcuda", "clang", "nvcc"):
        if (out_dir / f"{tag}.ptx").exists():
            result[tag] = metrics_for_tag(out_dir, tag)
    return result


def _print_table(out_dir):
    data = metrics_for_dir(out_dir)
    if not data:
        print(f"no artifacts found in {out_dir}")
        return
    keys = sorted({k for m in data.values() for k in m})
    tags = list(data)
    width = max(len(k) for k in keys) + 2
    header = "metric".ljust(width) + "".join(t.rjust(12) for t in tags)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = k.ljust(width) + "".join(str(data[t].get(k, "")).rjust(12) for t in tags)
        print(row)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/artifacts/vecadd"
    _print_table(target)
