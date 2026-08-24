// Same L1 tiling + dual-buffer annotation as the validated
// standalone/runtime/tile_l1_pipelined.mlir schedule, except the three tile
// sizes are transform.tune.knob ops instead of literals -- resolved by
// mlir_schedule.render_schedule() (see ../../mlir_schedule.py) before this
// file is ever fed to mlir-opt. `interchange = [2, 0, 1]` (K outermost) and
// `snitch.dual_buffer` stay fixed literals, not knobs: K must stay outermost
// and unpipelined, or dual-buffering the reduction loop silently drops data
// (see CLAUDE.md's Milestone 4 writeup, "The hardest bug: pipelining a
// reduction loop silently drops data").
//
// tile_m/tile_n/tile_k are NOT free integers -- see
// snitch_tuner/backend.py's snitch_tile_constraint() docstring for why
// tile_m must be a multiple of 8 (M6's barrier-participant symmetry
// depends on it), why all three must evenly divide the payload's M/N/K,
// and (found empirically, not predicted) why tile_n is effectively fixed
// at N/2 -- LowerPipelineOp's on-ramp/steady-state/off-ramp expansion only
// generalizes correctly to a pipeline trip count of exactly 2. The
// `options` lists below are informational only -- see
// mlir_schedule.render_schedule()'s docstring -- the tuner's real search
// space is defined entirely by snitch_tile_constraint/
// snitch_divisor_sample_value in Python, not by what's written here.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> = 16 from options = [16] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> = 16 from options = [16] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> = 32 from options = [32] -> !transform.param<i64>
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [%tile_m, %tile_n, %tile_k]
      interchange = [2, 0, 1]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.annotate %tiled "snitch.dual_buffer" : !transform.any_op
    transform.yield
  }
}
