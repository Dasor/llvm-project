// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(snitch-split-reduction-remainder))' --split-input-file | FileCheck %s

// K = 90 tiled by 24: the last iteration (k = 72) computes on the valid
// 18-wide part of its tiles, and the A/B padding is no longer zero-filled.

// CHECK-LABEL: func.func @k_remainder
// CHECK-DAG:     %[[POISON:.+]] = ub.poison : f64
// CHECK-DAG:     %[[C72:.+]] = arith.constant 72 : index
// CHECK:         scf.for %[[K:.+]] = {{.*}} iter_args(%[[ACC:.+]] = %{{.+}})
// CHECK:           %[[A:.+]] = tensor.pad
// CHECK:             tensor.yield %[[POISON]]
// CHECK:           %[[B:.+]] = tensor.pad
// CHECK:             tensor.yield %[[POISON]]
// CHECK:           %[[NOT_LAST:.+]] = arith.cmpi ne, %[[K]], %[[C72]]
// CHECK:           %[[R:.+]] = scf.if %[[NOT_LAST]]
// CHECK:             %[[FULL:.+]] = linalg.matmul {snitch.dual_buffer} ins(%[[A]], %[[B]] : tensor<32x24xf64>, tensor<24x32xf64>) outs(%[[ACC]]
// CHECK:             scf.yield %[[FULL]]
// CHECK:           } else {
// CHECK:             %[[AR:.+]] = tensor.extract_slice %[[A]][0, 0] [32, 18] [1, 1]
// CHECK:             %[[BR:.+]] = tensor.extract_slice %[[B]][0, 0] [18, 32] [1, 1]
// CHECK:             %[[REM:.+]] = linalg.matmul ins(%[[AR]], %[[BR]] : tensor<32x18xf64>, tensor<18x32xf64>) outs(%[[ACC]]
// CHECK:             scf.yield %[[REM]]
// CHECK:           scf.yield %[[R]]
#min = affine_map<(d0) -> (-d0 + 90, 24)>
#pad = affine_map<(d0) -> (-d0 + 24)>
func.func @k_remainder(%a: tensor<32x90xf64>, %b: tensor<90x32xf64>, %c: tensor<32x32xf64>) -> tensor<32x32xf64> {
  %cst = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c90 = arith.constant 90 : index
  %r = scf.for %k = %c0 to %c90 step %c24 iter_args(%acc = %c) -> (tensor<32x32xf64>) {
    %size = affine.min #min(%k)
    %high = affine.apply #pad(%size)
    %as = tensor.extract_slice %a[0, %k] [32, %size] [1, 1] : tensor<32x90xf64> to tensor<32x?xf64>
    %bs = tensor.extract_slice %b[%k, 0] [%size, 32] [1, 1] : tensor<90x32xf64> to tensor<?x32xf64>
    %ap = tensor.pad %as low[0, 0] high[0, %high] {
    ^bb0(%i: index, %j: index):
      tensor.yield %cst : f64
    } : tensor<32x?xf64> to tensor<32x24xf64>
    %bp = tensor.pad %bs low[0, 0] high[%high, 0] {
    ^bb0(%i: index, %j: index):
      tensor.yield %cst : f64
    } : tensor<?x32xf64> to tensor<24x32xf64>
    %m = linalg.matmul {snitch.dual_buffer} ins(%ap, %bp : tensor<32x24xf64>, tensor<24x32xf64>) outs(%acc : tensor<32x32xf64>) -> tensor<32x32xf64>
    scf.yield %m : tensor<32x32xf64>
  }
  return %r : tensor<32x32xf64>
}

// -----

// The step divides the trip range: nothing to split.

// CHECK-LABEL: func.func @divisible
// CHECK-NOT:     scf.if
// CHECK-NOT:     ub.poison
func.func @divisible(%a: tensor<32x96xf64>, %b: tensor<96x32xf64>, %c: tensor<32x32xf64>) -> tensor<32x32xf64> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c96 = arith.constant 96 : index
  %r = scf.for %k = %c0 to %c96 step %c24 iter_args(%acc = %c) -> (tensor<32x32xf64>) {
    %as = tensor.extract_slice %a[0, %k] [32, 24] [1, 1] : tensor<32x96xf64> to tensor<32x24xf64>
    %bs = tensor.extract_slice %b[%k, 0] [24, 32] [1, 1] : tensor<96x32xf64> to tensor<24x32xf64>
    %m = linalg.matmul ins(%as, %bs : tensor<32x24xf64>, tensor<24x32xf64>) outs(%acc : tensor<32x32xf64>) -> tensor<32x32xf64>
    scf.yield %m : tensor<32x32xf64>
  }
  return %r : tensor<32x32xf64>
}

// -----

// The loop tiles a parallel dimension (N), not the reduction: nothing to split.

// CHECK-LABEL: func.func @parallel_loop
// CHECK-NOT:     scf.if
#min = affine_map<(d0) -> (-d0 + 90, 24)>
#pad = affine_map<(d0) -> (-d0 + 24)>
func.func @parallel_loop(%a: tensor<32x24xf64>, %b: tensor<24x90xf64>, %c: tensor<32x96xf64>) -> tensor<32x96xf64> {
  %cst = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c90 = arith.constant 90 : index
  %r = scf.for %j = %c0 to %c90 step %c24 iter_args(%cc = %c) -> (tensor<32x96xf64>) {
    %size = affine.min #min(%j)
    %high = affine.apply #pad(%size)
    %bs = tensor.extract_slice %b[0, %j] [24, %size] [1, 1] : tensor<24x90xf64> to tensor<24x?xf64>
    %bp = tensor.pad %bs low[0, 0] high[0, %high] {
    ^bb0(%i: index, %k: index):
      tensor.yield %cst : f64
    } : tensor<24x?xf64> to tensor<24x24xf64>
    %cs = tensor.extract_slice %cc[0, %j] [32, 24] [1, 1] : tensor<32x96xf64> to tensor<32x24xf64>
    %m = linalg.matmul ins(%a, %bp : tensor<32x24xf64>, tensor<24x24xf64>) outs(%cs : tensor<32x24xf64>) -> tensor<32x24xf64>
    %ins = tensor.insert_slice %m into %cc[0, %j] [32, 24] [1, 1] : tensor<32x24xf64> into tensor<32x96xf64>
    scf.yield %ins : tensor<32x96xf64>
  }
  return %r : tensor<32x96xf64>
}
