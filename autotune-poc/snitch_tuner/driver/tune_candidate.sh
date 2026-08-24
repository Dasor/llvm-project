#!/usr/bin/env bash
set -euo pipefail

# Similar to snitch-backend/run_matmul32_multicore_pipelined_gvsoc.sh
# e2e compilation of a candidate schedule, run in gvsoc, check correctness.
# usage: tune_candidate.sh <resolved-schedule.mlir> <workdir>

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <resolved-schedule.mlir> <workdir>" >&2
  exit 2
fi
SCHEDULE="$1"
WORKDIR="$2"
mkdir -p "$WORKDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # llvm-project/ root
                                                    # (driver/ -> snitch_tuner/ -> autotune-poc/ -> here)
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # autotune-poc/snitch_tuner/

# Inside this LLVM checkout -- same binaries autotune-poc/driver/compile_schedule.sh uses.
LLVM_BUILD="${LLVM_BUILD_DIR:-$REPO_ROOT/build}"
MLIR_OPT="$LLVM_BUILD/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_BUILD/bin/mlir-translate"
LLC="$LLVM_BUILD/bin/llc"

SNITCH_ROOT="${SNITCH_ROOT:-/home/daniel/quidditch}"
XDSL_OPT="${XDSL_OPT:-$SNITCH_ROOT/venv/bin/xdsl-opt}"
GVSOC_ROOT="${GVSOC_ROOT:-$SNITCH_ROOT/gvsoc}"
GVSOC_PYTHON_ENV="${GVSOC_PYTHON_ENV:-/home/daniel/.conda/envs/gvsoc-py311}"
SNITCH_TOOLCHAIN="${SNITCH_TOOLCHAIN:-$SNITCH_ROOT/toolchain/bin}"
SNRUNTIME_API="${SNRUNTIME_API:-$SNITCH_ROOT/snitch_cluster/sw/snRuntime/api}"
RISCV_OPCODES_API="${RISCV_OPCODES_API:-$SNITCH_ROOT/snitch_cluster/sw/deps/riscv-opcodes}"
RTL_API="${RTL_API:-$SNITCH_ROOT/runtime/snitch_cluster/api}"
SNRUNTIME_BUILD="${SNRUNTIME_BUILD:-$SNITCH_ROOT/build/runtime}"   # contains snitch_cluster/libsnRuntime.a, base.ld
SNRUNTIME_LIB="${SNRUNTIME_LIB:-$SNITCH_ROOT/snitch_cluster/sw/snRuntime}"
SNRUNTIME_RTL="${SNRUNTIME_RTL:-$SNITCH_ROOT/runtime/snitch_cluster/rtl}"

PREFIX="$WORKDIR/m6"

echo "[+] Step 1a: tile via the transform dialect (resolved knobs)"
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$SCHEDULE" \
  --transform-interpreter --cse --canonicalize \
  "$POC_DIR/payload/matmul_32.mlir" \
  -o "${PREFIX}_step1a.mlir"

echo "[+] Step 1b: promote operands to L1, CSE, canonicalize"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-promote-operands-to-l1,cse,canonicalize))' \
  "${PREFIX}_step1a.mlir" -o "${PREFIX}_step1b.mlir"

echo "[+] Step 1c: create snitch pipeline"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-pipeline-copy-compute))' \
  "${PREFIX}_step1b.mlir" -o "${PREFIX}_step1c.mlir"

echo "[+] Step 1d: tile the compute stage into per-hart scf.forall (thread-tiling)"
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$POC_DIR/schedules/tile_l1_multicore.mlir" \
  --transform-interpreter --cse --canonicalize \
  "${PREFIX}_step1c.mlir" -o "${PREFIX}_step1d.mlir"

echo "[+] Step 1e: eliminate empty tensors, form microkernels"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-form-microkernels,snitch-eliminate-empty-tensors))' \
  "${PREFIX}_step1d.mlir" -o "${PREFIX}_step1.mlir"

echo "[+] Step 2a: bufferize (function boundaries + real DMA memcpy)"
"$MLIR_OPT" --snitch-bufferize='use-dma-memcpy=true' --cse --canonicalize \
  "${PREFIX}_step1.mlir" -o "${PREFIX}_step2.mlir"

echo "[+] Step 2b: expand the pipeline compute-stage occurrences into scf.forall"
"$MLIR_OPT" --snitch-lower-pipeline-op --cse --canonicalize \
  "${PREFIX}_step2.mlir" -o "${PREFIX}_step2b.mlir"

echo "[+] Step 2c: lower scf.forall to strided scf.for (one per hart)"
"$MLIR_OPT" --snitch-lower-forall-op='compute-cores=8' --cse --canonicalize \
  "${PREFIX}_step2b.mlir" -o "${PREFIX}_step2c.mlir"

echo "[+] Step 3: lower L1 allocations, specialize DM/compute-core, legalize DMA ops"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-lower-l1-allocations),cse,canonicalize,snitch-specialize-dma-code,func.func(snitch-dma-legalize-dma-operations,cse,canonicalize))' \
  "${PREFIX}_step2c.mlir" -o "${PREFIX}_step3.mlir"

echo "[+] Step 4: convert-to-riscv (xDSL microkernel handoff)"
"$MLIR_OPT" --linalg-generalize-named-ops \
  "${PREFIX}_step3.mlir" -o "${PREFIX}_step4_generalized.mlir"
"$MLIR_OPT" \
  --pass-pipeline="builtin.module(convert-to-riscv{xdsl-opt-path=$XDSL_OPT assert-compiled=true})" \
  "${PREFIX}_step4_generalized.mlir" -o "${PREFIX}_step5_riscv.mlir"

echo "[+] Step 5: lower Snitch/DMA/SCF/MemRef/Affine/Arith/Func to the LLVM dialect"
"$MLIR_OPT" \
  --convert-scf-to-cf \
  --convert-snitch-to-llvm='barrier-participants=9' \
  --convert-dma-to-llvm \
  --expand-strided-metadata \
  --lower-affine \
  --finalize-memref-to-llvm='index-bitwidth=32' \
  --convert-arith-to-llvm='index-bitwidth=32' \
  --convert-cf-to-llvm='index-bitwidth=32' \
  --convert-func-to-llvm='index-bitwidth=32 use-bare-ptr-memref-call-conv=true' \
  --reconcile-unrealized-casts \
  "${PREFIX}_step5_riscv.mlir" -o "${PREFIX}_step6_llvm.mlir"

echo "[+] Step 6: translate to LLVM IR and compile to a RISC-V object"
"$MLIR_TRANSLATE" --mlir-to-llvmir "${PREFIX}_step6_llvm.mlir" -o "${PREFIX}.ll"
"$LLC" -mtriple=riscv32-unknown-unknown-elf -mcpu=generic-rv32 -mattr=+m,+f,+d,+zfh -filetype=obj \
  -o "${PREFIX}_llvm.o" "${PREFIX}.ll"

echo "[+] Step 7: extract xDSL microkernel assembly and assemble"
rm -f "${PREFIX}"_xdsl_kernel*.S "${PREFIX}"_xdsl_kernel*.o
python3 "$SCRIPT_DIR/extract_asm.py" "${PREFIX}_step6_llvm.mlir" "${PREFIX}_xdsl_kernel"
KERNEL_OBJS=()
for s in "${PREFIX}"_xdsl_kernel*.S; do
  o="${s%.S}.o"
  "$SNITCH_TOOLCHAIN/pulp-as" --filetype=obj --target-abi=ilp32d "$s" \
    -o "$o" --mcpu=snitch -g
  KERNEL_OBJS+=("$o")
done

echo "[+] Step 8: merge the LLVM object with the xDSL kernel object(s)"
"$SNITCH_TOOLCHAIN/ld.lld" -r "${PREFIX}_llvm.o" "${KERNEL_OBJS[@]}" -o "${PREFIX}_kernel.o"

echo "[+] Step 9: compile the (cycle-instrumented) C harness"
"$SNITCH_TOOLCHAIN/clang" \
  -isystem "$SNRUNTIME_API" \
  -isystem "$RISCV_OPCODES_API" \
  -isystem "$RTL_API" \
  -g -O3 -DNDEBUG -std=gnu11 -Wno-undefined-inline \
  -o "${PREFIX}_main.c.obj" -c "$POC_DIR/runtime/main32_multicore_tune.c"

echo "[+] Step 10: link against libsnRuntime.a"
(
  cd "$SNRUNTIME_BUILD"
  "$SNITCH_TOOLCHAIN/clang" -g -O3 -DNDEBUG -lm -Tbase.ld \
    "${PREFIX}_main.c.obj" "${PREFIX}_kernel.o" -o "${PREFIX}_exe" \
    -L"$SNRUNTIME_LIB" \
    -L"$SNRUNTIME_RTL" \
    snitch_cluster/libsnRuntime.a
)
echo "Built: ${PREFIX}_exe"

echo "[+] Step 11: run on gvsoc"
GVSOC_WORK_DIR="$WORKDIR/gvsoc"
mkdir -p "$GVSOC_WORK_DIR"
export PATH="$GVSOC_PYTHON_ENV/bin:$GVSOC_ROOT/install/bin:$PATH"

GVSOC_LOG="$GVSOC_WORK_DIR/gvsoc_run.log"
set +e
gvrun --target snitch --work-dir "$GVSOC_WORK_DIR" --param chip/soc/binary="${PREFIX}_exe" run \
  | tee "$GVSOC_LOG"
GVRUN_STATUS=$?
set -e

if [ "$GVRUN_STATUS" -ne 0 ]; then
  echo "[!] gvrun exited with status $GVRUN_STATUS"
  exit "$GVRUN_STATUS"
fi

if ! grep -qE "all [0-9]+ values matched" "$GVSOC_LOG"; then
  echo "[!] Did not find 'all N values matched' in gvsoc output -- treating as failure"
  exit 1
fi

echo "[+] gvsoc run passed"
