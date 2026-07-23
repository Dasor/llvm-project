// Demonstrates transform.smt.constrain_params: the joint-constraint
// mechanism from slide p.11 of the deck (e.g. "tile_m * tile_n <= bound").
//
// IMPORTANT / finding from this checkout: transform.smt.constrain_params
// parses and verifies fine, but as of this LLVM revision it has NO
// interpreted semantics -- running it through `--transform-interpreter`
// fails with:
//   "op does not have interpreted semantics yet"
// (confirmed below; also: mlir/test/Dialect/Transform/test-smt-extension.mlir
// upstream only ever parses this op, it never runs it through
// --transform-interpreter either).
//
// Practical consequence for this proof-of-concept: the SMT op is currently
// a *declarative-only* constraint language -- useful for documenting/
// communicating the search-space constraint, or for a future/external
// constraint solver to consume, but NOT something that can be embedded in
// a schedule that actually gets executed today. That's why
// driver/autotune.py enforces the equivalent constraint in plain Python
// when enumerating candidates, and why schedules that actually run
// (knobs_resolved_valid.mlir, and everything driver/autotune.py renders)
// omit this op entirely.
//
// Run (parse-only, this succeeds):
//   mlir-opt schedules/smt_constraint_declaration.mlir
//
// Run (interpretation, this fails with "does not have interpreted
// semantics yet" -- left in on purpose to demonstrate the limitation):
//   mlir-opt schedules/smt_constraint_declaration.mlir --transform-interpreter

func.func @f(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %d = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%c : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %d : tensor<256x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> = 32 from options = [16, 32, 64] -> !transform.param<i64>

    // Joint constraint: tile_m * tile_n <= 4096.
    transform.smt.constrain_params(%tile_m, %tile_n) : (!transform.param<i64>, !transform.param<i64>) -> () {
      ^bb0(%m: !smt.int, %n: !smt.int):
      %prod = smt.int.mul %m, %n
      %bound = smt.int.constant 4096
      %ok = smt.int.cmp le %prod, %bound
      smt.assert %ok
    }

    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [%tile_m, %tile_n, %tile_k]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
