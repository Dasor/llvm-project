// Demonstrates failure mode 1: an unresolved transform.tune.knob.
//
// Run: mlir-opt knobs_unresolved.mlir --transform-interpreter
//
// Expected: a definite interpreter failure, NOT a crash --
//   "non-deterministic choice 'tile_m' is only resolved through providing
//    a `selected` attr"
// This is the core mechanism: a knob with only `options` (no `selected`)
// is a placeholder for a choice that has not been made yet. The interpreter
// refuses to guess.

func.func @f(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %d = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%c : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %d : tensor<256x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> options = [16, 32, 64] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> options = [16, 32, 64] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> options = [16, 32, 64] -> !transform.param<i64>
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [%tile_m, %tile_n, %tile_k]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
