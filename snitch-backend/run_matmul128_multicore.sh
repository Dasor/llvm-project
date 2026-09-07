#!/usr/bin/env bash
# Matmul 128x128 in Snitch backend, tiled 64x32x32 (M,N,K), multicore (all 8
# compute harts), NOT pipelined/dual-buffered, run on the cycle-accurate
# Verilator model (snitch_cluster_bin.vlt). Same pipeline as
# run_matmul128_multicore_pipelined.sh minus the pipelining steps -- a
# pipelined-vs-not cycle-count comparison at the same tile size. Ported from
# the older, non-containerized standalone/run_matmul32_multicore.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

die() { echo "[!] $*" >&2; exit 1; }

for f in "$MLIR_OPT" "$XDSL_OPT"; do
  [ -x "$f" ] || die "$f not found -- run ./build.sh first"
done
[ -f "$SNRT/lib/libsnRuntime.a" ] || die "$SNRT/lib/libsnRuntime.a not found -- run ./build.sh (--verilator) first"
[ -x "$SNITCH_CLUSTER_VLT" ] || die "$SNITCH_CLUSTER_VLT not found -- run ./build.sh --verilator first"

mkdir -p "$OUT"

# Step 1a: tile the matmul into L1-sized tiles (64x32x64), plain (no pipelining)
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$KERNELS/tile_l1_128.mlir" \
  --transform-interpreter --cse --canonicalize \
  "$KERNELS/matmul_128.mlir" \
  -o "$OUT/m128mc_step1a.mlir"

# Step 1a2: promote operands to L1 (must run before thread-tiling, so all 8
# cores share one DMA'd-in L1 tile instead of each getting its own)
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-promote-operands-to-l1,cse,canonicalize))' \
  "$OUT/m128mc_step1a.mlir" -o "$OUT/m128mc_step1a2.mlir"

# Step 1a3: thread-tile across 8 compute cores (tile_using_forall)
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$KERNELS/tile_l1_multicore.mlir" \
  --transform-interpreter --cse --canonicalize \
  "$OUT/m128mc_step1a2.mlir" -o "$OUT/m128mc_step1a3.mlir"

# Step 1b: form microkernels + eliminate empty tensors
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-form-microkernels,snitch-eliminate-empty-tensors))' \
  "$OUT/m128mc_step1a3.mlir" -o "$OUT/m128mc_step1.mlir"

# Step 2: bufferize (function boundaries + real DMA memcpy)
"$MLIR_OPT" --snitch-bufferize='use-dma-memcpy=true' --cse --canonicalize \
  "$OUT/m128mc_step1.mlir" -o "$OUT/m128mc_step2.mlir"

# Step 2c: lower scf.forall into a strided per-hart scf.for
# (must run before snitch-lower-l1-allocations)
"$MLIR_OPT" --snitch-lower-forall-op='compute-cores=8' --cse --canonicalize \
  "$OUT/m128mc_step2.mlir" -o "$OUT/m128mc_step2c.mlir"

# Step 3: lower L1 allocations, specialize DM/compute-core, legalize DMA ops
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-lower-l1-allocations),cse,canonicalize,snitch-specialize-dma-code,func.func(snitch-dma-legalize-dma-operations,cse,canonicalize))' \
  "$OUT/m128mc_step2c.mlir" -o "$OUT/m128mc_step3.mlir"

# Step 4: convert-to-riscv (xDSL microkernel handoff for the 8x32x32 per-hart tile)
"$MLIR_OPT" --linalg-generalize-named-ops \
  "$OUT/m128mc_step3.mlir" -o "$OUT/m128mc_step4_generalized.mlir"
"$MLIR_OPT" \
  --pass-pipeline="builtin.module(convert-to-riscv{xdsl-opt-path=$XDSL_OPT assert-compiled=true})" \
  "$OUT/m128mc_step4_generalized.mlir" -o "$OUT/m128mc_step5_riscv.mlir"

# Step 5: lower Snitch/DMA/SCF/MemRef/Affine/Arith/Func to the LLVM dialect
# (barrier-participants=9: 8 compute harts + 1 DM hart share every barrier)
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
  "$OUT/m128mc_step5_riscv.mlir" -o "$OUT/m128mc_step6_llvm.mlir"

# Step 6: translate to LLVM IR and compile to a RISC-V object via llc
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/m128mc_step6_llvm.mlir" -o "$OUT/m128mc.ll"
"$LLC" -mtriple=riscv32-unknown-unknown-elf -mcpu=generic-rv32 -mattr=+m,+f,+d,+zfh -filetype=obj \
  -o "$OUT/m128mc_llvm.o" "$OUT/m128mc.ll"

# Step 7: extract the xDSL-produced microkernel assembly and assemble it
rm -f "$OUT"/m128mc_xdsl_kernel*.S "$OUT"/m128mc_xdsl_kernel*.o
python3 "$SB_ROOT/extract_asm.py" "$OUT/m128mc_step6_llvm.mlir" "$OUT/m128mc_xdsl_kernel"
KERNEL_OBJS=()
for s in "$OUT"/m128mc_xdsl_kernel*.S; do
  o="${s%.S}.o"
  "$TOOLCHAIN/bin/pulp-as" --filetype=obj --target-abi=ilp32d "$s" -o "$o" --mcpu=snitch -g
  KERNEL_OBJS+=("$o")
done

# Step 8: merge all relocatable objects
"$TOOLCHAIN/bin/ld.lld" -r "$OUT/m128mc_llvm.o" "${KERNEL_OBJS[@]}" -o "$OUT/m128mc_kernel.o"

# Step 9: compile main32_multicore.c, sized to 128x128 via -D flags
"$TOOLCHAIN/bin/clang" \
  -isystem "$SNRT/include" \
  -DM=128 -DK=128 -DN=128 \
  -g -O3 -DNDEBUG -std=gnu11 -Wno-undefined-inline \
  -o "$OUT/main128mc.c.obj" -c "$KERNELS/main32_multicore.c"

# Step 10: link against libsnRuntime.a
"$TOOLCHAIN/bin/clang" -g -O3 -DNDEBUG -lm \
  -T "$SNRT/ld/base.ld" -L "$SNRT/ld" \
  "$OUT/main128mc.c.obj" "$OUT/m128mc_kernel.o" -o "$OUT/matmul128_multicore" \
  "$SNRT/lib/libsnRuntime.a"

# Step 11: run on the cycle-accurate Verilator model
VLT_WORK_DIR="$OUT/out_128_multicore_verilator"
mkdir -p "$VLT_WORK_DIR"
cd "$VLT_WORK_DIR"

VLT_LOG="$VLT_WORK_DIR/verilator_run.log"
set +e
"$SNITCH_CLUSTER_VLT" "$OUT/matmul128_multicore" | tee "$VLT_LOG"
VLT_STATUS=$?
set -e

if [ "$VLT_STATUS" -ne 0 ]; then
  echo "[!] Verilator run exited with status $VLT_STATUS"
  exit "$VLT_STATUS"
fi

if ! grep -qE "all [0-9]+ values matched" "$VLT_LOG"; then
  echo "[!] Did not find 'all N values matched' in Verilator output -- treating as failure"
  exit 1
fi

echo "[+] Verilator run passed: 128x128 tiled 64x32x32, all 8 compute cores, no pipelining (cycle-accurate)"
