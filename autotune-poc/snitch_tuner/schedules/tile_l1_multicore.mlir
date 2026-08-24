module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %mm num_threads [8, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
