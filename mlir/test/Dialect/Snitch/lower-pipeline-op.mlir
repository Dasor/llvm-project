// RUN: mlir-opt %s --snitch-lower-pipeline-op --allow-unregistered-dialect --split-input-file | FileCheck %s
// RUN: mlir-opt %s --snitch-lower-pipeline-op=prefetch-across-iterations=false --allow-unregistered-dialect --split-input-file | FileCheck %s --check-prefix=OFF

// The compute stage's waits for its own transfers come before the copy stage
// of the next iteration issues new transfers, in every step of the pipeline.

// CHECK-LABEL: func.func @waits_before_next_copies
// On-ramp: copy of iteration 0.
// CHECK:         %[[T0:.+]] = dma.start_transfer
// Steady state: wait for iteration i, then issue iteration i+1, then compute i.
// CHECK:         scf.for {{.*}} iter_args({{.*}}, %[[T:.+]] = %[[T0]])
// CHECK-NEXT:      affine.apply
// CHECK-NEXT:      affine.apply
// CHECK-NEXT:      dma.wait_for_transfer %[[T]]
// CHECK:           %[[TN:.+]] = dma.start_transfer
// CHECK:           "test.compute"
// CHECK:           scf.yield {{.*}}, %[[TN]]
// Off-ramp: wait and compute for the last iteration.
// CHECK:         dma.wait_for_transfer
// CHECK-NOT:     dma.start_transfer
// CHECK:         "test.compute"
func.func @waits_before_next_copies(%src: memref<128xf64>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  snitch.pipeline %c0 to %c128 step %c32 {
  ^bb0(%iv: index):
    %s = memref.subview %src[%iv] [32] [1] : memref<128xf64> to memref<32xf64, strided<[1], offset: ?>>
    %buf = memref.alloca() : memref<32xf64, #snitch.l1_encoding>
    %t = dma.start_transfer from %s : memref<32xf64, strided<[1], offset: ?>> to %buf : memref<32xf64, #snitch.l1_encoding>
    snitch.pipeline_yield %buf, %t : memref<32xf64, #snitch.l1_encoding>, !dma.token
  }, {
  ^bb0(%iv: index, %buf: memref<32xf64, #snitch.l1_encoding>, %t: !dma.token):
    dma.wait_for_transfer %t
    "test.compute"(%buf) : (memref<32xf64, #snitch.l1_encoding>) -> ()
  }
  return
}

// -----

// A pipeline that is the body of a loop issues the copy stage of the next
// iteration of the loop during its last compute stage. Its buffers are hoisted
// out of the loop into an alloca scope.

// CHECK-LABEL: func.func @prefetch_across_iterations
// CHECK:         memref.alloca_scope
// CHECK-COUNT-2:   memref.alloca()
// Copy stage of the first iteration of the loop, before it.
// CHECK:           %[[T0:.+]] = dma.start_transfer
// CHECK:           scf.for %[[J:.+]] = {{.*}} iter_args({{.*}}, %[[TJ:.+]] = %[[T0]])
// The first step of the pipeline waits for it.
// CHECK:             scf.for {{.*}} iter_args({{.*}}, %[[T:.+]] = %[[TJ]])
// CHECK:               dma.wait_for_transfer %[[T]]
// CHECK:               dma.start_transfer
// CHECK:               memref.copy
// CHECK:             }
// Off-ramp: wait for the last copy, prefetch for the next iteration, compute.
// CHECK:             dma.wait_for_transfer
// CHECK:             %[[NEXT:.+]] = arith.addi %[[J]]
// CHECK:             %[[COND:.+]] = arith.cmpi slt, %[[NEXT]]
// CHECK:             %[[TN:[0-9]+]]:2 = scf.if %[[COND]]
// CHECK:               %[[S:.+]] = memref.subview %{{.+}}[%[[NEXT]], %{{.+}}]
// CHECK:               %[[TP:.+]] = dma.start_transfer from %[[S]]
// CHECK:               scf.yield {{.*}}%[[TP]]
// CHECK:             } else {
// CHECK:               %[[DONE:.+]] = dma.completed_token
// CHECK:               scf.yield {{.*}}%[[DONE]]
// CHECK:             memref.copy
// CHECK:             scf.yield %[[TN]]#0, %[[TN]]#1
func.func @prefetch_across_iterations(%src: memref<4x128xf64>, %acc: memref<32xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.for %j = %c0 to %c4 step %c1 {
    snitch.pipeline %c0 to %c128 step %c32 {
    ^bb0(%iv: index):
      %s = memref.subview %src[%j, %iv] [1, 32] [1, 1] : memref<4x128xf64> to memref<32xf64, strided<[1], offset: ?>>
      %buf = memref.alloca() : memref<32xf64, #snitch.l1_encoding>
      %t = dma.start_transfer from %s : memref<32xf64, strided<[1], offset: ?>> to %buf : memref<32xf64, #snitch.l1_encoding>
      snitch.pipeline_yield %buf, %t : memref<32xf64, #snitch.l1_encoding>, !dma.token
    }, {
    ^bb0(%iv: index, %buf: memref<32xf64, #snitch.l1_encoding>, %t: !dma.token):
      dma.wait_for_transfer %t
      memref.copy %buf, %acc : memref<32xf64, #snitch.l1_encoding> to memref<32xf64>
    }
  }
  return
}

// -----

// Not with an odd number of pipeline iterations: the first and the last
// iteration use the same buffer.

// CHECK-LABEL: func.func @no_prefetch_odd_trip_count
// CHECK-NOT:     memref.alloca_scope
// CHECK:         scf.for
// CHECK-NOT:     scf.if
// CHECK:         return
func.func @no_prefetch_odd_trip_count(%src: memref<4x96xf64>, %acc: memref<32xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c96 = arith.constant 96 : index
  scf.for %j = %c0 to %c4 step %c1 {
    snitch.pipeline %c0 to %c96 step %c32 {
    ^bb0(%iv: index):
      %s = memref.subview %src[%j, %iv] [1, 32] [1, 1] : memref<4x96xf64> to memref<32xf64, strided<[1], offset: ?>>
      %buf = memref.alloca() : memref<32xf64, #snitch.l1_encoding>
      %t = dma.start_transfer from %s : memref<32xf64, strided<[1], offset: ?>> to %buf : memref<32xf64, #snitch.l1_encoding>
      snitch.pipeline_yield %buf, %t : memref<32xf64, #snitch.l1_encoding>, !dma.token
    }, {
    ^bb0(%iv: index, %buf: memref<32xf64, #snitch.l1_encoding>, %t: !dma.token):
      dma.wait_for_transfer %t
      memref.copy %buf, %acc : memref<32xf64, #snitch.l1_encoding> to memref<32xf64>
    }
  }
  return
}

// -----

// Not if the loop writes memory the copy stage reads: the copy for the next
// iteration would move ahead of the write.

// CHECK-LABEL: func.func @no_prefetch_loop_writes_source
// CHECK-NOT:     memref.alloca_scope
// CHECK-NOT:     scf.if
// CHECK:         return
func.func @no_prefetch_loop_writes_source(%src: memref<4x128xf64>, %acc: memref<32xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.for %j = %c0 to %c4 step %c1 {
    snitch.pipeline %c0 to %c128 step %c32 {
    ^bb0(%iv: index):
      %s = memref.subview %src[%j, %iv] [1, 32] [1, 1] : memref<4x128xf64> to memref<32xf64, strided<[1], offset: ?>>
      %buf = memref.alloca() : memref<32xf64, #snitch.l1_encoding>
      %t = dma.start_transfer from %s : memref<32xf64, strided<[1], offset: ?>> to %buf : memref<32xf64, #snitch.l1_encoding>
      snitch.pipeline_yield %buf, %t : memref<32xf64, #snitch.l1_encoding>, !dma.token
    }, {
    ^bb0(%iv: index, %buf: memref<32xf64, #snitch.l1_encoding>, %t: !dma.token):
      dma.wait_for_transfer %t
      memref.copy %buf, %acc : memref<32xf64, #snitch.l1_encoding> to memref<32xf64>
    }
    %next = memref.subview %src[%j, 0] [1, 32] [1, 1] : memref<4x128xf64> to memref<32xf64, strided<[1], offset: ?>>
    %w = dma.start_transfer from %acc : memref<32xf64> to %next : memref<32xf64, strided<[1], offset: ?>>
    dma.wait_for_transfer %w
  }
  return
}

// OFF-LABEL: func.func @prefetch_across_iterations
// OFF-NOT:     memref.alloca_scope
// OFF-NOT:     scf.if
// OFF:         return
