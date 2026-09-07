module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // 128x128x128 case, no pipelining: plain 64x32x32 (M,N,K) tiling, default loop order.
    // Matches tile_l1_pipelined_128.mlir's tile size (64x32x32, not the originally
    // requested 64x32x64 -- see that file's comment) so the pipelined-vs-not
    // comparison between run_matmul128_multicore.sh and
    // run_matmul128_multicore_pipelined.sh is apples-to-apples at the same tile size.
    // This script's build is single-buffered and would have had headroom for 64x32x64
    // on its own (full ~110KiB available, not the ~80KiB dual-buffered budget).
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [64, 32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
