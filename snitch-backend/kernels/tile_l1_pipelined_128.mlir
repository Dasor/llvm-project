module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // 128x128x128 case: tile 64x32x32 (M,N,K). Requested as 64x32x64, but that
    // doesn't fit dual-buffered -- lhs+rhs+out at 64x32x64 needs 128KiB, and the L1
    // scratch view is only ~110KiB (confirmed directly: snitch-lower-l1-allocations
    // emits "kernel does not fit into L1 memory and cannot be compiled" for 64x32x64
    // here). 64x32x32 fits with real margin (~80KiB of ~110KiB) and keeps two of the
    // three requested dimensions. Same interchange as tile_l1_pipelined.mlir (K
    // outermost, N innermost) -- current limitation of snitch-pipeline-copy-compute
    // inherited from quidditch: it can't safely pipeline the K/reduction loop.
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [64, 32, 32]
      interchange = [2, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Mark the tiled matmul for double buffering.
    transform.annotate %tiled "snitch.dual_buffer" : !transform.any_op
    transform.yield
  }
}
