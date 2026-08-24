module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> options = [8, 16, 24, 32] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> options = [12, 20, 24] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> options = [12, 20, 24] -> !transform.param<i64>
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [%tile_m, %tile_n, %tile_k]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %padded, %pad, %copy = transform.structured.pad %tiled {padding_values = [0.0 : f64, 0.0 : f64, 0.0 : f64], padding_dimensions = [0, 1, 2]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
