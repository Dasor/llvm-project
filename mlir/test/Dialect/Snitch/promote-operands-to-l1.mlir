// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(snitch-promote-operands-to-l1))' --split-input-file | FileCheck %s

// A loop-carried accumulator is copied into L1 once, before the loop; the
// per-iteration operands are copied inside it.

// CHECK-LABEL: func.func @loop_carried
// CHECK-SAME:    %[[CS:[^:]+]]: tensor<32x64xf64>
// CHECK:         %[[CL1:.+]], %[[T:.+]] = dma.start_tensor_copy of %[[CS]]
// CHECK:         %[[W:.+]] = dma.wait_for_tensor_copy of %[[CS]]
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[W]])
// CHECK-COUNT-2:   dma.start_tensor_copy
// CHECK-NOT:       dma.start_tensor_copy
// CHECK:           linalg.matmul {{.*}} outs(%[[ACC]] : tensor<32x64xf64>)
func.func @loop_carried(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %cs: tensor<32x64xf64>, %i: index, %j: index) -> tensor<32x64xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %k = %c0 to %c128 step %c32 iter_args(%acc = %cs) -> (tensor<32x64xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %m = linalg.matmul ins(%as, %bs : tensor<32x32xf64>, tensor<32x64xf64>) outs(%acc : tensor<32x64xf64>) -> tensor<32x64xf64>
    scf.yield %m : tensor<32x64xf64>
  }
  return %r : tensor<32x64xf64>
}

// -----

// An accumulator already in L1 (e.g. a promoted pad) is not copied again.

// CHECK-LABEL: func.func @already_l1
// CHECK:         dma.start_tensor_copy
// CHECK:         scf.for
// CHECK-COUNT-2:   dma.start_tensor_copy
// CHECK-NOT:       dma.start_tensor_copy
// CHECK:           linalg.matmul
func.func @already_l1(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %cs: tensor<32x64xf64>, %i: index, %j: index) -> tensor<32x64xf64> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %l1, %tok = dma.start_tensor_copy of %cs to #snitch.l1_encoding -> tensor<32x64xf64>
  %w = dma.wait_for_tensor_copy of %cs : tensor<32x64xf64> to %l1 using %tok -> tensor<32x64xf64>
  %r = scf.for %k = %c0 to %c128 step %c32 iter_args(%acc = %w) -> (tensor<32x64xf64>) {
    %as = tensor.extract_slice %a[%i, %k] [32, 32] [1, 1] : tensor<128x128xf64> to tensor<32x32xf64>
    %bs = tensor.extract_slice %b[%k, %j] [32, 64] [1, 1] : tensor<128x128xf64> to tensor<32x64xf64>
    %m = linalg.matmul ins(%as, %bs : tensor<32x32xf64>, tensor<32x64xf64>) outs(%acc : tensor<32x64xf64>) -> tensor<32x64xf64>
    scf.yield %m : tensor<32x64xf64>
  }
  return %r : tensor<32x64xf64>
}
