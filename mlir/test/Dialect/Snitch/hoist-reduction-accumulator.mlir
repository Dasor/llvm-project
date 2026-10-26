// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(snitch-hoist-reduction-accumulator,cse,canonicalize))' --split-input-file | FileCheck %s

// Divisible tile: the C slice and the per-iteration copy-back leave the loop,
// which carries the C tile itself.

// CHECK-LABEL: func.func @divisible
// CHECK-SAME:    %[[A:[^:]+]]: tensor<128x128xf64>, %[[B:[^:]+]]: tensor<128x128xf64>, %[[C:[^:]+]]: tensor<128x128xf64>
// CHECK:         %[[CS:.+]] = tensor.extract_slice %[[C]][%{{.+}}, %{{.+}}] [32, 64] [1, 1]
// CHECK:         %[[R:.+]] = scf.for %{{.+}} iter_args(%[[ACC:.+]] = %[[CS]]) -> (tensor<32x64xf64>)
// CHECK-NOT:       bufferization.materialize_in_destination
// CHECK:           %[[M:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xf64>, tensor<32x64xf64>) outs(%[[ACC]] : tensor<32x64xf64>)
// CHECK:           scf.yield %[[M]]
// CHECK:         tensor.insert_slice %[[R]] into %[[C]]
func.func @divisible(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %c: tensor<128x128xf64>, %i: index, %j: index) -> tensor<128x128xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %k = %c0 to %c128 step %c32 iter_args(%cc = %c) -> (tensor<128x128xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %cs = tensor.extract_slice %cc[%i, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %m = linalg.matmul ins(%as, %bs : tensor<32x32xf64>, tensor<32x64xf64>) outs(%cs : tensor<32x64xf64>) -> tensor<32x64xf64>
    %w = bufferization.materialize_in_destination %m in %cs : (tensor<32x64xf64>, tensor<32x64xf64>) -> tensor<32x64xf64>
    %ins = tensor.insert_slice %w into %cc[%i, %j] [32, 64] [1, 1] : tensor<32x64xf64> into tensor<128x128xf64>
    scf.yield %ins : tensor<128x128xf64>
  }
  return %r : tensor<128x128xf64>
}

// -----

// Padded (non-divisible) tile: the loop carries the padded tile; the pad moves
// before the loop and the unpadding slice after it.

#min = affine_map<(d0) -> (-d0 + 100, 32)>
#minK = affine_map<(d0) -> (-d0 + 90, 32)>
#pad = affine_map<(d0) -> (-d0 + 32)>

// CHECK-LABEL: func.func @padded
// CHECK-SAME:    %[[C:[^:]+]]: tensor<100x72xf64>
// CHECK:         %[[CS:.+]] = tensor.extract_slice %[[C]]
// CHECK:         %[[P:.+]] = tensor.pad %[[CS]] low[0, 0]
// CHECK:         %[[R:.+]] = scf.for %{{.+}} iter_args(%[[ACC:.+]] = %[[P]]) -> (tensor<32x32xf64>)
// CHECK-COUNT-2:   tensor.pad
// CHECK:           %[[M:.+]] = linalg.matmul {{.*}} outs(%[[ACC]] : tensor<32x32xf64>)
// CHECK-NEXT:      scf.yield %[[M]]
// CHECK:         %[[U:.+]] = tensor.extract_slice %[[R]][0, 0]
// CHECK:         tensor.insert_slice %[[U]] into %[[C]]
func.func @padded(%a: tensor<100x90xf64>, %b: tensor<90x72xf64>, %c: tensor<100x72xf64>, %i: index, %j: index) -> tensor<100x72xf64> {
  %cst = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c90 = arith.constant 90 : index
  %r = scf.for %k = %c0 to %c90 step %c32 iter_args(%cc = %c) -> (tensor<100x72xf64>) {
    %sm = affine.min #min(%i)
    %sn = affine.min #min(%j)
    %sk = affine.min #minK(%k)
    %as = tensor.extract_slice %a[%i, %k] [%sm, %sk] [1, 1] : tensor<100x90xf64> to tensor<?x?xf64>
    %bs = tensor.extract_slice %b[%k, %j] [%sk, %sn] [1, 1] : tensor<90x72xf64> to tensor<?x?xf64>
    %cs = tensor.extract_slice %cc[%i, %j] [%sm, %sn] [1, 1] : tensor<100x72xf64> to tensor<?x?xf64>
    %hm = affine.apply #pad(%sm)
    %hn = affine.apply #pad(%sn)
    %hk = affine.apply #pad(%sk)
    %ap = tensor.pad %as low[0, 0] high[%hm, %hk] {
    ^bb0(%x: index, %y: index):
      tensor.yield %cst : f64
    } : tensor<?x?xf64> to tensor<32x32xf64>
    %bp = tensor.pad %bs low[0, 0] high[%hk, %hn] {
    ^bb0(%x: index, %y: index):
      tensor.yield %cst : f64
    } : tensor<?x?xf64> to tensor<32x32xf64>
    %cp = tensor.pad %cs low[0, 0] high[%hm, %hn] {
    ^bb0(%x: index, %y: index):
      tensor.yield %cst : f64
    } : tensor<?x?xf64> to tensor<32x32xf64>
    %m = linalg.matmul {snitch.dual_buffer} ins(%ap, %bp : tensor<32x32xf64>, tensor<32x32xf64>) outs(%cp : tensor<32x32xf64>) -> tensor<32x32xf64>
    %u = tensor.extract_slice %m[0, 0] [%sm, %sn] [1, 1] : tensor<32x32xf64> to tensor<?x?xf64>
    %ins = tensor.insert_slice %u into %cc[%i, %j] [%sm, %sn] [1, 1] : tensor<?x?xf64> into tensor<100x72xf64>
    scf.yield %ins : tensor<100x72xf64>
  }
  return %r : tensor<100x72xf64>
}

// -----

// The C slice depends on the loop's induction variable (a parallel, not a
// reduction, loop): nothing is hoisted.

// CHECK-LABEL: func.func @not_invariant
// CHECK:         scf.for %[[J:.+]] = {{.*}} iter_args(%[[CC:.+]] = %{{.+}}) -> (tensor<128x128xf64>)
// CHECK:           %[[CS:.+]] = tensor.extract_slice %[[CC]][%{{.+}}, %[[J]]] [32, 32]
// CHECK:           linalg.matmul {{.*}} outs(%[[CS]] : tensor<32x32xf64>)
// CHECK:           tensor.insert_slice
func.func @not_invariant(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %c: tensor<128x128xf64>, %i: index) -> tensor<128x128xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %j = %c0 to %c128 step %c32 iter_args(%cc = %c) -> (tensor<128x128xf64>) {
    %as = tensor.extract_slice %a[%i, 0] [32, 128] [1, 1] : tensor<128x128xf64> to tensor<32x128xf64>
    %bs = tensor.extract_slice %b[0, %j] [128, 32] [1, 1] : tensor<128x128xf64> to tensor<128x32xf64>
    %cs = tensor.extract_slice %cc[%i, %j] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %m = linalg.matmul ins(%as, %bs : tensor<32x128xf64>, tensor<128x32xf64>) outs(%cs : tensor<32x32xf64>) -> tensor<32x32xf64>
    %ins = tensor.insert_slice %m into %cc[%i, %j] [32, 32] [1, 1] : tensor<32x32xf64> into tensor<128x128xf64>
    scf.yield %ins : tensor<128x128xf64>
  }
  return %r : tensor<128x128xf64>
}
