#!/usr/bin/env bash
# Ahead-of-time compiles a fully-resolved transform-dialect schedule (payload
# + schedule in one .mlir file, all transform.tune.knob ops must already have
# a `selected` value) down to a standalone native executable. Does NOT run
# it -- mlir_schedule.py's run_candidate() times execution separately, so
# compilation (this script) is never part of the measured interval. See
# README.md's timing note for why that matters.
#
# Same mlir-opt pass pipeline as run_one.sh (the JIT-based reference path
# used for the non-timing-sensitive Step 2/3 demonstrations), just piped
# through mlir-translate + llc + a system C compiler instead of mlir-runner.
#
# Usage: compile_schedule.sh <schedule.mlir> <output-executable-path>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${LLVM_BUILD_DIR:-$REPO_ROOT/build}"

MLIR_OPT="$BUILD_DIR/bin/mlir-opt"
MLIR_TRANSLATE="$BUILD_DIR/bin/mlir-translate"
LLC="$BUILD_DIR/bin/llc"
CC="${CC:-cc}"
C_RUNNER_UTILS="$BUILD_DIR/lib/libmlir_c_runner_utils.so"
RUNNER_UTILS="$BUILD_DIR/lib/libmlir_runner_utils.so"
LIBOMP="${LIBOMP:-/lib64/libomp.so}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <schedule.mlir> <output-executable-path>" >&2
  exit 2
fi
SCHEDULE="$1"
OUT_EXE="$2"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

"$MLIR_OPT" "$SCHEDULE" -transform-interpreter \
  -test-transform-dialect-erase-schedule \
  -canonicalize -cse \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline -convert-bufferization-to-memref \
  -scf-forall-to-parallel -convert-scf-to-openmp -canonicalize \
  -convert-linalg-to-loops \
  -convert-vector-to-scf="full-unroll" \
  -lower-vector-mask -lower-vector-multi-reduction \
  -convert-scf-to-cf -expand-strided-metadata -lower-affine \
  -convert-arith-to-llvm \
  -convert-vector-to-llvm="vector-contract-lowering=outerproduct" \
  -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm \
  -convert-openmp-to-llvm -convert-ub-to-llvm -reconcile-unrealized-casts \
  -canonicalize -cse \
| "$MLIR_TRANSLATE" --mlir-to-llvmir -o "$WORKDIR/out.ll"

"$LLC" -filetype=obj -mcpu=native "$WORKDIR/out.ll" -o "$WORKDIR/out.o"

"$CC" "$WORKDIR/out.o" -o "$OUT_EXE" \
  "$C_RUNNER_UTILS" "$RUNNER_UTILS" "$LIBOMP" -Wl,-rpath,"$BUILD_DIR/lib"
