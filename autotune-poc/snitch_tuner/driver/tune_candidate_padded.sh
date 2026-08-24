#!/usr/bin/env bash
set -euo pipefail

# usage: tune_candidate_padded.sh <resolved-schedule.mlir> <payload.mlir> <workdir>

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <resolved-schedule.mlir> <payload.mlir> <workdir>" >&2
  exit 2
fi
SCHEDULE="$1"
PAYLOAD="$2"
WORKDIR="$3"
mkdir -p "$WORKDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # llvm-project/ root
                                                    # (driver/ -> snitch_tuner/ -> autotune-poc/ -> here)
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # autotune-poc/snitch_tuner/

# M/N/K come from the payload's own linalg.matmul op, not a hardcoded
# constant -- reuses mlir_schedule.detect_matmul_dims (the exact same
# regex the Python side already uses) so there's only one place that
# parses this, not a second shell implementation that could drift.
read -r DIM_M DIM_N DIM_K <<< "$(python3 -c "
import sys
sys.path.insert(0, '$POC_DIR/..')
from mlir_schedule import detect_matmul_dims
d = detect_matmul_dims(open('$PAYLOAD').read())
print(d['M'], d['N'], d['K'])
")"
echo "[+] Payload dims: M=$DIM_M N=$DIM_N K=$DIM_K"

# Inside this LLVM checkout -- same binaries autotune-poc/driver/compile_schedule.sh uses.
LLVM_BUILD="${LLVM_BUILD_DIR:-$REPO_ROOT/build}"
MLIR_OPT="$LLVM_BUILD/bin/mlir-opt"
MLIR_TRANSLATE="$LLVM_BUILD/bin/mlir-translate"
LLC="$LLVM_BUILD/bin/llc"

# Genuinely external (not part of any LLVM checkout) -- overridable, defaults
# match this machine's current setup, same pattern compile_schedule.sh uses
# for CC/LIBOMP.
# Self-contained: everything below lives inside llvm-project/snitch-backend/
# (see its env.sh and runtime/snruntime/PROVENANCE.md), not in a sibling
# Quidditch checkout. SB_ROOT can still be overridden if a different
# snitch-backend checkout should be used.
SB_ROOT="${SB_ROOT:-$REPO_ROOT/snitch-backend}"
VENV="${VENV:-$SB_ROOT/build/.venv}"
XDSL_OPT="${XDSL_OPT:-$VENV/bin/xdsl-opt}"
GVSOC_WORKDIR="${GVSOC_WORKDIR:-$SB_ROOT/build/gvsoc}"
SNITCH_TOOLCHAIN="${SNITCH_TOOLCHAIN:-${QUIDDITCH_TOOLCHAIN:-$SB_ROOT/toolchain}/bin}"
SNRT="${SNRT:-$SB_ROOT/runtime/snruntime}"
SNRUNTIME_API="${SNRUNTIME_API:-$SNRT/include}"
RISCV_OPCODES_API="${RISCV_OPCODES_API:-$SB_ROOT/runtime/riscv-opcodes}"

PREFIX="$WORKDIR/m5p"

echo "[+] Step 1a: tile + pad via the transform dialect (resolved knobs)"
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$SCHEDULE" \
  --transform-interpreter --cse --canonicalize \
  "$PAYLOAD" \
  -o "${PREFIX}_step1a.mlir"

echo "[+] Step 1b: promote pads + operands to L1, CSE, canonicalize"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-promote-pads-to-l1,cse,canonicalize,snitch-promote-operands-to-l1,cse,canonicalize))' \
  "${PREFIX}_step1a.mlir" -o "${PREFIX}_step1b.mlir"

echo "[+] Step 1d: tile the compute into per-hart scf.forall (thread-tiling)"
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$POC_DIR/schedules/tile_l1_multicore.mlir" \
  --transform-interpreter --cse --canonicalize \
  "${PREFIX}_step1b.mlir" -o "${PREFIX}_step1d.mlir"

echo "[+] Step 1e: eliminate empty tensors, form microkernels"
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-form-microkernels,snitch-eliminate-empty-tensors))' \
  "${PREFIX}_step1d.mlir" -o "${PREFIX}_step1.mlir"

echo "[+] Step 2a: bufferize (function boundaries + real DMA memcpy)"
"$MLIR_OPT" --snitch-bufferize='use-dma-memcpy=true' --cse --canonicalize \
  "${PREFIX}_step1.mlir" -o "${PREFIX}_step2.mlir"

echo "[+] Step 2c: lower scf.forall to strided scf.for (one per hart)"
"$MLIR_OPT" --snitch-lower-forall-op='compute-cores=8' --cse --canonicalize \
  "${PREFIX}_step2.mlir" -o "${PREFIX}_step2c.mlir"

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
  -DM=$DIM_M -DN=$DIM_N -DK=$DIM_K \
  -g -O3 -DNDEBUG -std=gnu11 -Wno-undefined-inline \
  -o "${PREFIX}_main.c.obj" -c "$POC_DIR/runtime/main32_multicore_tune.c"

echo "[+] Step 10: link against libsnRuntime.a"
"$SNITCH_TOOLCHAIN/clang" -g -O3 -DNDEBUG -lm \
  -T"$SNRT/ld/base.ld" -L"$SNRT/ld" \
  "${PREFIX}_main.c.obj" "${PREFIX}_kernel.o" -o "${PREFIX}_exe" \
  "$SNRT/lib/libsnRuntime.a"
echo "Built: ${PREFIX}_exe"

echo "[+] Step 11: run on gvsoc"
GVSOC_WORK_DIR="$WORKDIR/gvsoc"
mkdir -p "$GVSOC_WORK_DIR"
export PATH="$VENV/bin:$GVSOC_WORKDIR/install/bin:$PATH"

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
