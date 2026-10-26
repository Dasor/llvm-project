// RUN: mlir-opt %s --snitch-specialize-dma-code | FileCheck %s

// Waits for several transfers in a row share one barrier.

// CHECK-LABEL: func.func @consecutive_waits(
// CHECK:         snitch.barrier
// CHECK-NOT:     snitch.barrier
// CHECK:         return

// CHECK-LABEL: func.func @consecutive_waits$dma
// CHECK:         dma.start_transfer
// CHECK:         dma.start_transfer
// CHECK:         dma.wait_for_transfer
// CHECK-NEXT:    dma.wait_for_transfer
// CHECK-NEXT:    snitch.barrier
// CHECK-NOT:     snitch.barrier
// CHECK:         return
func.func @consecutive_waits(%a: memref<32xf64>, %b: memref<32xf64>, %la: memref<32xf64>, %lb: memref<32xf64>) {
  %ta = dma.start_transfer from %a : memref<32xf64> to %la : memref<32xf64>
  %tb = dma.start_transfer from %b : memref<32xf64> to %lb : memref<32xf64>
  dma.wait_for_transfer %ta
  dma.wait_for_transfer %tb
  return
}
