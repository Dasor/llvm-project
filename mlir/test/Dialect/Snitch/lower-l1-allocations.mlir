// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(snitch-lower-l1-allocations))' --split-input-file --allow-unregistered-dialect 2>&1 | FileCheck %s

// Allocations in loop nests that run one after another (e.g. the nests of a
// tile loop and its peeled remainder tile) share L1 memory. An allocation
// defined after a nest may reuse the nest's memory; one defined before it may
// not.

// The remark for @escaping below; diagnostics are printed before the IR.
// CHECK:       remark: L1 allocation may outlive its block

// CHECK-LABEL: func.func @sequential_nests
// CHECK:         %[[L1:.+]] = snitch.l1_memory_view
// Live across both nests: offset 0.
// CHECK:         memref.view %[[L1]][%c0{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<128xf64>
// First nest: above the live buffer.
// CHECK:         scf.for
// CHECK:           memref.view %[[L1]][%c1024{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<256xf64>
// Second nest: reuses the first nest's memory.
// CHECK:         scf.for
// CHECK:           memref.view %[[L1]][%c1024{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<64xf64>
// Defined after both nests: reuses their memory too.
// CHECK:         memref.view %[[L1]][%c1024{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<32xf64>
func.func @sequential_nests(%lb: index, %ub: index, %step: index) {
  %live = memref.alloca() {alignment = 64 : i64} : memref<128xf64, #snitch.l1_encoding>
  "test.use"(%live) : (memref<128xf64, #snitch.l1_encoding>) -> ()
  scf.for %i = %lb to %ub step %step {
    %a = memref.alloca() {alignment = 64 : i64} : memref<256xf64, #snitch.l1_encoding>
    "test.use"(%a, %live) : (memref<256xf64, #snitch.l1_encoding>, memref<128xf64, #snitch.l1_encoding>) -> ()
  }
  scf.for %i = %lb to %ub step %step {
    %b = memref.alloca() {alignment = 64 : i64} : memref<64xf64, #snitch.l1_encoding>
    "test.use"(%b, %live) : (memref<64xf64, #snitch.l1_encoding>, memref<128xf64, #snitch.l1_encoding>) -> ()
  }
  %after = memref.alloca() {alignment = 64 : i64} : memref<32xf64, #snitch.l1_encoding>
  "test.use"(%after, %live) : (memref<32xf64, #snitch.l1_encoding>, memref<128xf64, #snitch.l1_encoding>) -> ()
  return
}

// -----

// An allocation whose memory may outlive its block (yielded by the block's
// terminator) disables reuse: every allocation gets its own memory.

// CHECK-LABEL: func.func @escaping
// CHECK:         memref.view %{{.+}}[%c0{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<256xf64>
// CHECK:         memref.view %{{.+}}[%c2048{{(_[0-9]+)?}}][] : memref<112640xi8> to memref<256xf64>
func.func @escaping(%c: i1) {
  %r = scf.if %c -> memref<256xf64, #snitch.l1_encoding> {
    %a = memref.alloca() {alignment = 64 : i64} : memref<256xf64, #snitch.l1_encoding>
    scf.yield %a : memref<256xf64, #snitch.l1_encoding>
  } else {
    %b = memref.alloca() {alignment = 64 : i64} : memref<256xf64, #snitch.l1_encoding>
    scf.yield %b : memref<256xf64, #snitch.l1_encoding>
  }
  "test.use"(%r) : (memref<256xf64, #snitch.l1_encoding>) -> ()
  return
}

// -----

// An alloca scope limits the lifetime of the L1 allocations in it: an
// allocation after it reuses their memory. The scope is inlined once its
// allocations are lowered.

// CHECK-LABEL: func.func @alloca_scope
// CHECK-NOT:     memref.alloca_scope
// CHECK:         %[[OFF0:.+]] = arith.constant 0 : index
// CHECK:         memref.view %{{.+}}[%[[OFF0]]][]
// CHECK:         "test.use"
// CHECK:         %[[OFF1:.+]] = arith.constant 0 : index
// CHECK:         memref.view %{{.+}}[%[[OFF1]]][]
// CHECK:         "test.use"
func.func @alloca_scope() {
  memref.alloca_scope {
    %a = memref.alloca() : memref<64xf64, #snitch.l1_encoding>
    "test.use"(%a) : (memref<64xf64, #snitch.l1_encoding>) -> ()
  }
  %b = memref.alloca() : memref<64xf64, #snitch.l1_encoding>
  "test.use"(%b) : (memref<64xf64, #snitch.l1_encoding>) -> ()
  return
}
