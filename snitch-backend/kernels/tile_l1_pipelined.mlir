module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // tile 16x16x16 and interchange as current limitation of snitch-pipeline-copy-compute inherited from quidditch
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [16, 16, 16]
      interchange = [2, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Mark the tiled matmul for double buffering.
    transform.annotate %tiled "snitch.dual_buffer" : !transform.any_op
    transform.yield
  }
}
