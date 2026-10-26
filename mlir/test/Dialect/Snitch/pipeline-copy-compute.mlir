// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(snitch-pipeline-copy-compute))' --split-input-file --verify-diagnostics | FileCheck %s

// Reduction loop with the accumulator carried in L1: only the A and B copies
// move to the copy stage; the accumulator flows to the compute stage.

// CHECK-LABEL: func.func @hoisted_accumulator
// CHECK:         snitch.pipeline {{.*}} inits(%{{.+}}) -> tensor<32x64xf64> {
// CHECK-COUNT-2:   dma.start_tensor_copy
// CHECK:           snitch.pipeline_yield
// CHECK:         }, {
// CHECK-NOT:       dma.start_tensor_copy
// CHECK:           linalg.matmul
// CHECK:           snitch.pipeline_yield
func.func @hoisted_accumulator(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %cl1: tensor<32x64xf64>, %i: index, %j: index) -> tensor<32x64xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %k = %c0 to %c128 step %c32 iter_args(%acc = %cl1) -> (tensor<32x64xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %al1, %at = dma.start_tensor_copy of %as to #snitch.l1_encoding -> tensor<32x32xf64>
    %aw = dma.wait_for_tensor_copy of %as : tensor<32x32xf64> to %al1 using %at -> tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %bl1, %bt = dma.start_tensor_copy of %bs to #snitch.l1_encoding -> tensor<32x64xf64>
    %bw = dma.wait_for_tensor_copy of %bs : tensor<32x64xf64> to %bl1 using %bt -> tensor<32x64xf64>
    %m = linalg.matmul {snitch.dual_buffer} ins(%aw, %bw : tensor<32x32xf64>, tensor<32x64xf64>) outs(%acc : tensor<32x64xf64>) -> tensor<32x64xf64>
    scf.yield %m : tensor<32x64xf64>
  }
  return %r : tensor<32x64xf64>
}

// -----

// The C tile is copied from the loop-carried tensor on every iteration. The
// copy of iteration k+1 would run concurrently with the computation of
// iteration k, so the loop is not pipelined.

// CHECK-LABEL: func.func @loop_carried_copy
// CHECK-NOT:     snitch.pipeline
// CHECK:         scf.for
func.func @loop_carried_copy(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %c: tensor<128x128xf64>, %i: index, %j: index) -> tensor<128x128xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  // expected-warning @below {{not pipelining loop: a copy depends on a loop-carried value}}
  %r = scf.for %k = %c0 to %c128 step %c32 iter_args(%cc = %c) -> (tensor<128x128xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %al1, %at = dma.start_tensor_copy of %as to #snitch.l1_encoding -> tensor<32x32xf64>
    %aw = dma.wait_for_tensor_copy of %as : tensor<32x32xf64> to %al1 using %at -> tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %bl1, %bt = dma.start_tensor_copy of %bs to #snitch.l1_encoding -> tensor<32x64xf64>
    %bw = dma.wait_for_tensor_copy of %bs : tensor<32x64xf64> to %bl1 using %bt -> tensor<32x64xf64>
    // expected-note @below {{copy computed here}}
    %cs = tensor.extract_slice %cc[%i, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %cl1, %ct = dma.start_tensor_copy of %cs to #snitch.l1_encoding -> tensor<32x64xf64>
    %cw = dma.wait_for_tensor_copy of %cs : tensor<32x64xf64> to %cl1 using %ct -> tensor<32x64xf64>
    %m = linalg.matmul {snitch.dual_buffer} ins(%aw, %bw : tensor<32x32xf64>, tensor<32x64xf64>) outs(%cw : tensor<32x64xf64>) -> tensor<32x64xf64>
    %ins = tensor.insert_slice %m into %cc[%i, %j] [32, 64] [1, 1] : tensor<32x64xf64> into tensor<128x128xf64>
    scf.yield %ins : tensor<128x128xf64>
  }
  return %r : tensor<128x128xf64>
}

// -----

// Parallel loop: each iteration copies in and writes back its own C tile, so
// the copy of iteration j+1 does not read what iteration j writes. Pipelined.

// CHECK-LABEL: func.func @per_iteration_tile
// CHECK:         snitch.pipeline
// CHECK-COUNT-3:   dma.start_tensor_copy
// CHECK:           snitch.pipeline_yield
// CHECK:         }, {
// CHECK:           linalg.matmul
// CHECK:           tensor.insert_slice
func.func @per_iteration_tile(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %c: tensor<128x128xf64>, %i: index, %k: index) -> tensor<128x128xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %j = %c0 to %c128 step %c32 iter_args(%cc = %c) -> (tensor<128x128xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %al1, %at = dma.start_tensor_copy of %as to #snitch.l1_encoding -> tensor<32x32xf64>
    %aw = dma.wait_for_tensor_copy of %as : tensor<32x32xf64> to %al1 using %at -> tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %bl1, %bt = dma.start_tensor_copy of %bs to #snitch.l1_encoding -> tensor<32x32xf64>
    %bw = dma.wait_for_tensor_copy of %bs : tensor<32x32xf64> to %bl1 using %bt -> tensor<32x32xf64>
    %cs = tensor.extract_slice %cc[%i, %j] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %cl1, %ct = dma.start_tensor_copy of %cs to #snitch.l1_encoding -> tensor<32x32xf64>
    %cw = dma.wait_for_tensor_copy of %cs : tensor<32x32xf64> to %cl1 using %ct -> tensor<32x32xf64>
    %m = linalg.matmul {snitch.dual_buffer} ins(%aw, %bw : tensor<32x32xf64>, tensor<32x32xf64>) outs(%cw : tensor<32x32xf64>) -> tensor<32x32xf64>
    %ins = tensor.insert_slice %m into %cc[%i, %j] [32, 32] [1, 1] : tensor<32x32xf64> into tensor<128x128xf64>
    scf.yield %ins : tensor<128x128xf64>
  }
  return %r : tensor<128x128xf64>
}

// -----

// K remainder split off by snitch-split-reduction-remainder: the compute ops
// sit in an 'scf.if', the remainder one on slices of the copied tiles. The
// copies still move to the copy stage, the slices stay with the compute.

// CHECK-LABEL: func.func @split_remainder
// CHECK:         snitch.pipeline {{.*}} inits(%{{.+}}) -> tensor<32x32xf64> {
// CHECK-COUNT-2:   dma.start_tensor_copy
// CHECK:           snitch.pipeline_yield
// CHECK:         }, {
// CHECK-NOT:       dma.start_tensor_copy
// CHECK:           scf.if
// CHECK:             linalg.matmul {snitch.dual_buffer}
// CHECK:           } else {
// CHECK-COUNT-2:     tensor.extract_slice
// CHECK:             linalg.matmul
// CHECK:           snitch.pipeline_yield
#min = affine_map<(d0) -> (-d0 + 90, 24)>
func.func @split_remainder(%a: tensor<32x90xf64>, %b: tensor<90x32xf64>, %cl1: tensor<32x32xf64>) -> tensor<32x32xf64> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c72 = arith.constant 72 : index
  %c90 = arith.constant 90 : index
  %r = scf.for %k = %c0 to %c90 step %c24 iter_args(%acc = %cl1) -> (tensor<32x32xf64>) {
    %size = affine.min #min(%k)
    %high = affine.apply affine_map<(d0) -> (-d0 + 24)>(%size)
    %as = tensor.extract_slice %a[0, %k] [32, %size] [1, 1] : tensor<32x90xf64> to tensor<32x?xf64>
    %al1, %at = dma.start_tensor_copy of %as to #snitch.l1_encoding pad with undef by [0, %high] : tensor<32x?xf64> -> tensor<32x24xf64>
    %aw = dma.wait_for_tensor_copy of %as : tensor<32x?xf64> to %al1 using %at -> tensor<32x24xf64>
    %bs = tensor.extract_slice %b[%k, 0] [%size, 32] [1, 1] : tensor<90x32xf64> to tensor<?x32xf64>
    %bl1, %bt = dma.start_tensor_copy of %bs to #snitch.l1_encoding pad with undef by [%high, 0] : tensor<?x32xf64> -> tensor<24x32xf64>
    %bw = dma.wait_for_tensor_copy of %bs : tensor<?x32xf64> to %bl1 using %bt -> tensor<24x32xf64>
    %not_last = arith.cmpi ne, %k, %c72 : index
    %m = scf.if %not_last -> (tensor<32x32xf64>) {
      %full = linalg.matmul {snitch.dual_buffer} ins(%aw, %bw : tensor<32x24xf64>, tensor<24x32xf64>) outs(%acc : tensor<32x32xf64>) -> tensor<32x32xf64>
      scf.yield %full : tensor<32x32xf64>
    } else {
      %ar = tensor.extract_slice %aw[0, 0] [32, 18] [1, 1] : tensor<32x24xf64> to tensor<32x18xf64>
      %br = tensor.extract_slice %bw[0, 0] [18, 32] [1, 1] : tensor<24x32xf64> to tensor<18x32xf64>
      %rem = linalg.matmul ins(%ar, %br : tensor<32x18xf64>, tensor<18x32xf64>) outs(%acc : tensor<32x32xf64>) -> tensor<32x32xf64>
      scf.yield %rem : tensor<32x32xf64>
    }
    scf.yield %m : tensor<32x32xf64>
  }
  return %r : tensor<32x32xf64>
}
