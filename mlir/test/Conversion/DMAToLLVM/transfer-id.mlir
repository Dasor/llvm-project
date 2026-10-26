// RUN: mlir-opt %s --convert-dma-to-llvm | FileCheck %s

// The ID 'snrt_dma_start_*' returns is stale when the iDMA request FIFO is
// backed up. Every transfer gets max(returned ID, previous ID + 1), with the
// previous ID kept in a stack slot of the function.

// CHECK-LABEL: func.func @two_transfers
// CHECK:         %[[ONE:.+]] = llvm.mlir.constant(1 : i32)
// CHECK:         %[[SLOT:.+]] = llvm.alloca %[[ONE]] x i32 {dma.last_transfer_id}
// CHECK:         %[[ZERO:.+]] = llvm.mlir.constant(0 : i32)
// CHECK:         llvm.store %[[ZERO]], %[[SLOT]]
// CHECK:         %[[R0:.+]] = llvm.call @snrt_dma_start_1d
// CHECK:         %[[L0:.+]] = llvm.load %[[SLOT]]
// CHECK:         %[[N0:.+]] = llvm.add %[[L0]]
// CHECK:         %[[ID0:.+]] = llvm.intr.umax(%[[R0]], %[[N0]])
// CHECK:         llvm.store %[[ID0]], %[[SLOT]]
// CHECK:         %[[R1:.+]] = llvm.call @snrt_dma_start_1d
// CHECK:         %[[L1:.+]] = llvm.load %[[SLOT]]
// CHECK:         %[[N1:.+]] = llvm.add %[[L1]]
// CHECK:         %[[ID1:.+]] = llvm.intr.umax(%[[R1]], %[[N1]])
// CHECK:         llvm.store %[[ID1]], %[[SLOT]]
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.icmp "ult" %{{.+}}, %[[ID1]]
func.func @two_transfers(%a: memref<32xf64>, %b: memref<32xf64>,
                         %c: memref<32xf64>, %d: memref<32xf64>) {
  %t0 = dma.start_transfer from %a : memref<32xf64> to %b : memref<32xf64>
  %t1 = dma.start_transfer from %c : memref<32xf64> to %d : memref<32xf64>
  dma.wait_for_transfer %t1
  return
}
