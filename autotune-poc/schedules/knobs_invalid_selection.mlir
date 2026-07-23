// Demonstrates failure mode 2: a `selected` value that isn't a member of
// `options`.
//
// Run: mlir-opt knobs_invalid_selection.mlir --transform-interpreter
//
// Expected: a verifier error at parse time (before interpretation even
// starts) --
//   "'transform.tune.knob' op provided `selected` attribute is not an
//    element of `options` array of attributes"
// This is the guardrail that stops a search driver from silently feeding
// an out-of-space value into the schedule.

func.func @f(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %d = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%c : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %d : tensor<256x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // 17 is not in the options list below -- this must fail to verify.
    %tile_m = transform.tune.knob<"tile_m"> = 17 from options = [16, 32, 64] -> !transform.param<i64>
    %tiled, %l0 = transform.structured.tile_using_for %mm tile_sizes [%tile_m]
      : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
