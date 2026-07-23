#!/usr/bin/env bash
# Runs a single, fully-resolved transform-dialect schedule (payload +
# schedule in one .mlir file, all transform.tune.knob ops must already have
# a `selected` value) through the CPU lowering pipeline and JIT-executes it.
#
# This is the exact pipeline validated by hand in step 2/3 of the PoC
# (see ../schedules/baseline_fixed_tiles.mlir and
# ../schedules/knobs_resolved_valid.mlir), reused verbatim from
# mlir/test/Integration/Dialect/Linalg/CPU/test-tensor-matmul.mlir.
#
# Usage: run_one.sh <schedule.mlir>
# Prints whatever the payload's @main prints (e.g. printMemrefF32 output)
# to stdout. Exits non-zero if any pipeline stage fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${LLVM_BUILD_DIR:-$REPO_ROOT/build}"

MLIR_OPT="$BUILD_DIR/bin/mlir-opt"
MLIR_RUNNER="$BUILD_DIR/bin/mlir-runner"
C_RUNNER_UTILS="$BUILD_DIR/lib/libmlir_c_runner_utils.so"
RUNNER_UTILS="$BUILD_DIR/lib/libmlir_runner_utils.so"
LIBOMP="${LIBOMP:-/lib64/libomp.so}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <schedule.mlir>" >&2
  exit 2
fi
SCHEDULE="$1"

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
| "$MLIR_RUNNER" -e main -entry-point-result=void \
  -shared-libs="$C_RUNNER_UTILS,$RUNNER_UTILS,$LIBOMP"
