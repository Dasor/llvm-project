#!/usr/bin/env bash
# Shared path/tool resolution for build.sh and the run_*.sh drivers.
# Everything below derives from this file's own location, so nothing here
# is hardcoded to any particular host path or mount point.

SB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_ROOT="$(cd "$SB_ROOT/.." && pwd)"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_ROOT/build}"

VENV="$SB_ROOT/build/.venv"
GVSOC_ROOT="$SB_ROOT/gvsoc"
GVSOC_WORKDIR="$SB_ROOT/build/gvsoc"
TOOLCHAIN="${QUIDDITCH_TOOLCHAIN:-$SB_ROOT/toolchain}"
SNRT="$SB_ROOT/runtime/snruntime"
RISCV_OPCODES="$SB_ROOT/runtime/riscv-opcodes"
KERNELS="$SB_ROOT/kernels"
OUT="${OUT:-$SB_ROOT/out}"

SNITCH_CLUSTER_ROOT="$SB_ROOT/snitch_cluster"
SNITCH_CLUSTER_VENV="$SNITCH_CLUSTER_ROOT/.venv"
SNITCH_CLUSTER_VLT="$SNITCH_CLUSTER_ROOT/target/sim/build/bin/snitch_cluster_bin.vlt"

MLIR_OPT="$LLVM_BUILD/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_BUILD/bin/mlir-translate"
LLC="$LLVM_BUILD/bin/llc"
XDSL_OPT="$VENV/bin/xdsl-opt"

export UV_PYTHON_INSTALL_DIR="$SB_ROOT/build/.uv-python"
export UV_CACHE_DIR="$SB_ROOT/build/.uv-cache"

export PATH="$VENV/bin:$GVSOC_WORKDIR/install/bin:$PATH"

export SB_ROOT LLVM_ROOT LLVM_BUILD VENV GVSOC_ROOT GVSOC_WORKDIR TOOLCHAIN SNRT RISCV_OPCODES KERNELS OUT
export SNITCH_CLUSTER_ROOT SNITCH_CLUSTER_VENV SNITCH_CLUSTER_VLT
export MLIR_OPT MLIR_TRANSLATE LLC XDSL_OPT
