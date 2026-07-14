#!/usr/bin/env bash
# setup.sh — provision a Lightning AI Studio for the rust-cuda-vs-oxide study.
#
# Reproduces the toolchain via CONDA. The apt/LLVM route does NOT work on the
# Lightning image (no usable apt LLVM-21); conda-forge provides LLVM 21 with the
# NVPTX backend, and the nvidia channel provides the CUDA toolkit. Run on the
# free CPU Studio — only `cargo oxide run` needs a GPU. Re-runnable.
#
# Validated on Lightning's "cloudspace" conda env with a CUDA 12.8 PyTorch.
#
# NOTE: this sets up the oxide, clang, and nvcc cells. The rust-cuda cell needs
# a separate toolchain (LLVM 7 + libNVVM + a pinned nightly + a glam feature
# fix); its PTX is committed under results/artifacts/, and each rust-cuda/
# project documents its own build.

set -uo pipefail
OXIDE_NIGHTLY="nightly-2026-04-03"
say() { printf '\n=== %s ===\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

CONDA_ENV="${CONDA_PREFIX:?activate the conda env first: conda activate cloudspace}"
say "conda env: $CONDA_ENV"

CUDA_VER="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
CUDA_VER="${CUDA_VER:-12.8}"
say "1. CUDA toolkit $CUDA_VER (nvcc / ptxas / cuobjdump / headers + libcuda stub)"
have nvcc || conda install -y -c nvidia "cuda-toolkit=$CUDA_VER"

say "2. LLVM 21 + clang + libclang"
{ have llc && llc --version | grep -qi nvptx; } || \
  conda install -y -c conda-forge clang=21 clangxx=21 libclang=21 llvmdev=21

say "3. Rust $OXIDE_NIGHTLY + components"
if ! have rustup; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install "$OXIDE_NIGHTLY"
rustup component add rust-src rustc-dev llvm-tools --toolchain "$OXIDE_NIGHTLY"

say "4. Environment"
CUDA_HOME="$CONDA_ENV/targets/x86_64-linux"
export CUDA_HOME CUDA_TOOLKIT_PATH="$CUDA_HOME" CUDA_PATH="$CUDA_HOME"
export LIBCLANG_PATH="$CONDA_ENV/lib"
export CUDA_OXIDE_LLC="$CONDA_ENV/bin/llc"
export LIBRARY_PATH="$CUDA_HOME/lib/stubs:$CONDA_ENV/lib"
export BINDGEN_EXTRA_CLANG_ARGS="-I$CUDA_HOME/include"
mkdir -p "$HOME/.cudastub"
ln -sf "$CUDA_HOME/lib/stubs/libcuda.so" "$HOME/.cudastub/libcuda.so.1"
export LD_LIBRARY_PATH="$HOME/.cudastub:${LD_LIBRARY_PATH:-}"

MARK="# >>> rust-cuda-vs-oxide env >>>"
if ! grep -qF "$MARK" "$HOME/.zshenv" 2>/dev/null; then
  {
    echo "$MARK"
    echo "export CONDA_ENV=$CONDA_ENV"
    echo 'export PATH="$CONDA_ENV/bin:$HOME/.cargo/bin:$PATH"'
    echo 'export CUDA_HOME=$CONDA_ENV/targets/x86_64-linux'
    echo 'export CUDA_TOOLKIT_PATH=$CUDA_HOME'
    echo 'export CUDA_PATH=$CUDA_HOME'
    echo 'export LIBCLANG_PATH=$CONDA_ENV/lib'
    echo 'export CUDA_OXIDE_LLC=$CONDA_ENV/bin/llc'
    echo 'export LIBRARY_PATH=$CUDA_HOME/lib/stubs:$CONDA_ENV/lib'
    echo 'export BINDGEN_EXTRA_CLANG_ARGS="-I$CUDA_HOME/include"'
    echo 'export LD_LIBRARY_PATH=$HOME/.cudastub:${LD_LIBRARY_PATH:-}'
    echo "# <<< rust-cuda-vs-oxide env <<<"
  } >> "$HOME/.zshenv"
  echo "  wrote env block to ~/.zshenv"
else
  echo "  ~/.zshenv already has the env block — skipping"
fi

say "5. cargo-oxide"
have cargo-oxide || cargo "+$OXIDE_NIGHTLY" install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

say "6. Python deps"
pip install -r requirements.txt --quiet --break-system-packages 2>/dev/null || \
  pip install -r requirements.txt --quiet

say "7. Smoke test"
if ( cd kernels/vecadd && cargo oxide build --arch sm_89 ); then
  say "OK — environment ready. Next: python3 -m harness.build_all --config configs/vecadd.yaml --no-run"
else
  say "Smoke test failed — check the output above"
fi
