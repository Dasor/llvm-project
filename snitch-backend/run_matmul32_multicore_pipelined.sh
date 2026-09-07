#!/usr/bin/env bash
# Matmul 32x32 in Snitch backend, multicore, pipelined, dual-buffered, run on
# the cycle-accurate Verilator model (snitch_cluster_bin.vlt). Simulator-agnostic
# steps 1-10 are identical to run_matmul32_multicore_pipelined_gvsoc.sh -- only
# step 11 (which simulator runs the final ELF) differs.
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

# Step 1a: tile the matmul into L1-sized tiles (N-tile) and mark for dual-buffering
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$KERNELS/tile_l1_pipelined.mlir" \
  --transform-interpreter --cse --canonicalize \
  "$KERNELS/matmul_32.mlir" \
  -o "$OUT/m32mcp_step1a.mlir"

# Step 1b: promote operands to L1, CSE, canonicalize
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-promote-operands-to-l1,cse,canonicalize))' \
  "$OUT/m32mcp_step1a.mlir" -o "$OUT/m32mcp_step1b.mlir"

# Step 1c: Create snitch pipeline
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-pipeline-copy-compute))' \
  "$OUT/m32mcp_step1b.mlir" -o "$OUT/m32mcp_step1c.mlir"

# Step 1d: tile the compute stage into per-hart scf.forall (thread-tiling)
"$MLIR_OPT" \
  --transform-preload-library="transform-library-paths=$KERNELS/tile_l1_multicore.mlir" \
  --transform-interpreter --cse --canonicalize \
  "$OUT/m32mcp_step1c.mlir" -o "$OUT/m32mcp_step1d.mlir"

# Step 1e: eliminate empty tensors (from the thread-tiling) and snitch-form-microkernels
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-form-microkernels,snitch-eliminate-empty-tensors))' \
  "$OUT/m32mcp_step1d.mlir" -o "$OUT/m32mcp_step1.mlir"

# Step 2a: bufferize (with DMA memcpy) and CSE/canonicalize
"$MLIR_OPT" --snitch-bufferize='use-dma-memcpy=true' --cse --canonicalize \
  "$OUT/m32mcp_step1.mlir" -o "$OUT/m32mcp_step2.mlir"

# Step 2b: expand the pipeline compute-stage occurrences into scf.forall
"$MLIR_OPT" --snitch-lower-pipeline-op --cse --canonicalize \
  "$OUT/m32mcp_step2.mlir" -o "$OUT/m32mcp_step2b.mlir"

# Step 2c: lower the scf.forall into strided scf.for loops (one per hart)
"$MLIR_OPT" --snitch-lower-forall-op='compute-cores=8' --cse --canonicalize \
  "$OUT/m32mcp_step2b.mlir" -o "$OUT/m32mcp_step2c.mlir"

# Step 3: lower L1 allocations, CSE, canonicalize, specialize DMA code, legalize DMA ops
"$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(snitch-lower-l1-allocations),cse,canonicalize,snitch-specialize-dma-code,func.func(snitch-dma-legalize-dma-operations,cse,canonicalize))' \
  "$OUT/m32mcp_step2c.mlir" -o "$OUT/m32mcp_step3.mlir"

# Step 4: generalize named linalg ops (for xDSL compilation)
"$MLIR_OPT" --linalg-generalize-named-ops \
  "$OUT/m32mcp_step3.mlir" -o "$OUT/m32mcp_step4_generalized.mlir"
"$MLIR_OPT" \
  --pass-pipeline="builtin.module(convert-to-riscv{xdsl-opt-path=$XDSL_OPT assert-compiled=true})" \
  "$OUT/m32mcp_step4_generalized.mlir" -o "$OUT/m32mcp_step5_riscv.mlir"

# Step 5: lower to LLVM dialect (with xDSL's Snitch lowering)
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
  "$OUT/m32mcp_step5_riscv.mlir" -o "$OUT/m32mcp_step6_llvm.mlir"

# Step 6: convert to LLVM IR and compile to relocatable object
"$MLIR_TRANSLATE" --mlir-to-llvmir "$OUT/m32mcp_step6_llvm.mlir" -o "$OUT/m32mcp.ll"
"$LLC" -mtriple=riscv32-unknown-unknown-elf -mcpu=generic-rv32 -mattr=+m,+f,+d,+zfh -filetype=obj \
  -o "$OUT/m32mcp_llvm.o" "$OUT/m32mcp.ll"

# Step 7: extract the xDSL kernel assembly and compile to relocatable objects
rm -f "$OUT"/m32mcp_xdsl_kernel*.S "$OUT"/m32mcp_xdsl_kernel*.o
python3 "$SB_ROOT/extract_asm.py" "$OUT/m32mcp_step6_llvm.mlir" "$OUT/m32mcp_xdsl_kernel"
KERNEL_OBJS=()
for s in "$OUT"/m32mcp_xdsl_kernel*.S; do
  o="${s%.S}.o"
  "$TOOLCHAIN/bin/pulp-as" --filetype=obj --target-abi=ilp32d "$s" \
    -o "$o" --mcpu=snitch -g
  KERNEL_OBJS+=("$o")
done

# Step 8: link the xDSL kernel objects with the LLVM object into a single relocatable object
"$TOOLCHAIN/bin/ld.lld" -r "$OUT/m32mcp_llvm.o" "${KERNEL_OBJS[@]}" -o "$OUT/m32mcp_kernel.o"

# Step 9: compile the main32_multicore.c file into a relocatable object
"$TOOLCHAIN/bin/clang" \
  -isystem "$SNRT/include" \
  -g -O3 -DNDEBUG -std=gnu11 -Wno-undefined-inline \
  -o "$OUT/main32mcp.c.obj" -c "$KERNELS/main32_multicore.c"

# Step 10: link the main32_multicore.c object with the kernel object and the snRuntime library into a final executable
"$TOOLCHAIN/bin/clang" -g -O3 -DNDEBUG -lm \
  -T "$SNRT/ld/base.ld" -L "$SNRT/ld" \
  "$OUT/main32mcp.c.obj" "$OUT/m32mcp_kernel.o" -o "$OUT/matmul32_multicore_pipelined" \
  "$SNRT/lib/libsnRuntime.a"

# Step 11: run the final executable on the cycle-accurate Verilator model
VLT_WORK_DIR="$OUT/out_multicore_pipelined_verilator"
mkdir -p "$VLT_WORK_DIR"
cd "$VLT_WORK_DIR"

VLT_LOG="$VLT_WORK_DIR/verilator_run.log"
set +e
"$SNITCH_CLUSTER_VLT" "$OUT/matmul32_multicore_pipelined" | tee "$VLT_LOG"
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

echo "[+] Verilator run passed: dual-buffered pipelining + all 8 compute cores, combined (cycle-accurate)"
