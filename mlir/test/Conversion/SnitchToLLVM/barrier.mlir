// RUN: mlir-opt %s --convert-snitch-to-llvm=barrier-participants=9 | FileCheck %s --check-prefix=SW
// RUN: mlir-opt %s --convert-snitch-to-llvm="barrier-participants=9 hw-barrier=true" | FileCheck %s --check-prefix=HW

// SW-LABEL: func.func @barrier
// SW:         %[[BARRIER:.+]] = llvm.mlir.addressof @_snrt_barrier
// SW:         %[[N:.+]] = llvm.mlir.constant(9 : i32)
// SW:         llvm.call @snrt_partial_barrier(%[[BARRIER]], %[[N]])

// HW-LABEL: func.func @barrier
// HW-NOT:     snrt_partial_barrier
// HW:         llvm.inline_asm has_side_effects "csrr x0, 0x7c2", "~{memory}"
func.func @barrier() {
  snitch.barrier
  return
}
