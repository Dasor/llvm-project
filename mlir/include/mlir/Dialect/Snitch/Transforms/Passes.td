#ifndef MLIR_DIALECT_SNITCH_TRANSFORMS_PASSES
#define MLIR_DIALECT_SNITCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormMicrokernelsPass
  : InterfacePass<"snitch-form-microkernels", "mlir::FunctionOpInterface"> {
  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
  ];
}

def HoistReductionAccumulatorPass : Pass<"snitch-hoist-reduction-accumulator"> {
  let description = [{
    Keeps the accumulator of a tiled compute op live across the innermost
    tiling loop instead of extracting (and padding) it from the full output
    tensor and inserting it back on every iteration.

    Runs upstream loop-invariant code motion and subset hoisting on the
    `scf.for` directly surrounding each tensor-semantics linalg op, then
    rewrites the hoisted accumulator so the loop carries the (padded) tensor
    the op accumulates into: the `tensor.pad` moves before the loop, the
    unpadding `tensor.extract_slice` after it, and a per-iteration
    `bufferization.materialize_in_destination` copy-back is dropped.

    Run before L1 promotion: the accumulator is then copied into L1 once per
    loop instead of once per iteration, and the loop carries no data
    dependence through its copies, allowing `snitch-pipeline-copy-compute` to
    pipeline reduction loops.
  }];

  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
}

def PromotePadsToL1Pass : Pass<"snitch-promote-pads-to-l1"> {
  let description = [{
    Converts supported `tensor.pad` operations to `start_tensor_transfer` and
    `wait_for_tensor_copy` pairs.
  }];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
    "mlir::dma::DMADialect",
  ];
}

def PromoteOperandsToL1Pass : Pass<"snitch-promote-operands-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
    "mlir::dma::DMADialect",
  ];
}

def PromoteAllocsToL1Pass : Pass<"snitch-promote-allocs-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
  ];
}

def LowerL1AllocationsPass : InterfacePass<"snitch-lower-l1-allocations",
  "mlir::FunctionOpInterface"> {
  let description = [{

  }];

  let options = [
    Option<"l1MemoryBytes", "l1-memory-bytes", "unsigned", "112640", "">,
    Option<"assertCompiled", "assert-compiled", "bool", "false",
           "If true, errors if the kernel does not fit into L1 meory."
           "Otherwise, removes the kernel from the output and emits a warning "
           "instead.">,
  ];
}

def SpecializeDMACodePass : Pass<"snitch-specialize-dma-code",
  "mlir::ModuleOp"> {

  let description = [{
    Pass performing code specialization for DMA and compute cores while
    inserting required synchronization primitives.

    Every function originally present in the IR will be cloned and turned into
    a "dma" version.
    DMA versions have all compute operations (i.e. `mmemref.microkernel`s)
    removed while the original version has all DMA transfer operations removed.
    Barriers are inserted where data dependencies require either transfers or
    computations to have finished.
  }];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
  ];
}

def EliminateEmptyTensorsPass
  : InterfacePass<"snitch-eliminate-empty-tensors", "mlir::FunctionOpInterface"> {
  let description = [{
    Rewrites `tensor.empty` ops that are anchored on a destination-passing-style
    consumer to reuse that consumer's destination tensor instead, ahead of
    bufferization.
  }];

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
  ];
}

def BufferizePass : Pass<"snitch-bufferize", "mlir::ModuleOp"> {
  let description = [{
    Bufferizes the module using upstream One-Shot Bufferize, including
    function boundaries (with a static identity layout, since the exported
    functions must remain callable with a plain C ABI / xDSL's bare-pointer
    calling convention).

    When `use-dma-memcpy` is set, buffers are allocated with `memref.alloca`
    (suitable for the small, statically sized L1 scratchpad) and copies between
    buffers are lowered to `dma.start_transfer`/`dma.wait_for_transfer` pairs
    instead of the default `memref.alloc`/`linalg.copy`.
  }];

  let options = [
    Option<"useDMAMemcpy", "use-dma-memcpy", "bool", "false",
           "Allocate with memref.alloca and copy via the DMA dialect instead "
           "of the upstream defaults.">,
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::dma::DMADialect",
    "mlir::scf::SCFDialect",
  ];
}

def PipelineCopyComputePass : Pass<"snitch-pipeline-copy-compute"> {
  let description = [{
    Finds `scf.for` loops whose tiled compute op is marked with the
    `snitch.dual_buffer` unit attribute (attached by the tiling schedule,
    e.g. via `transform.annotate`) and lifts them into a two-stage
    `snitch.pipeline` (copy stage, compute stage) so a later
    `snitch-lower-pipeline-op` can expand it into a software-pipelined
    on-ramp/steady-state/off-ramp loop nest. Operates at the tensor level,
    before bufferization.
  }];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
    "mlir::dma::DMADialect",
  ];
}

def LowerForallOpPass : Pass<"snitch-lower-forall-op"> {
  let description = [{
    Converts a single-induction-variable `scf.forall` (as produced by
    thread-level tiling, e.g. `transform.structured.tile_using_forall`) into
    a strided `scf.for` where hart `id` processes iterations
    `id, id+compute_cores, id+2*compute_cores, ...`, using
    `snitch.compute_core_index` to determine `id`. Operates at the memref
    level, after bufferization.
  }];

  let options = [
    Option<"computeCores", "compute-cores", "unsigned", "8",
           "Number of compute cores (harts) to distribute forall iterations "
           "across.">,
  ];

  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
  ];
}

def LowerPipelineOpPass : Pass<"snitch-lower-pipeline-op"> {
  let description = [{
    Expands memref-semantics `snitch.pipeline` ops into the concrete
    on-ramp/steady-state/off-ramp `scf.for` nest, duplicating any
    `memref.alloca`s used by more than one stage so consecutive iterations
    of different stages can execute concurrently. Tensor-semantics
    `snitch.pipeline` ops (not yet bufferized) are left untouched. Operates
    at the memref level, after `snitch-lower-l1-allocations`.

    With `prefetch-across-iterations`, a two-stage pipeline that is the body
    of an enclosing `scf.for` (e.g. the K pipeline of an output tile in the
    loop over output tiles) issues the copy stage of the next iteration of
    that loop during its own last compute stage, instead of in the on-ramp of
    the next iteration: the first transfers of the next iteration then
    overlap with compute. The pipeline's buffers are hoisted out of the loop
    into a `memref.alloca_scope`. Only applied when legal: the first and last
    pipeline iterations use different buffers (an even trip count for two
    stages), the copy stage only starts transfers and depends on nothing in
    the loop but pure ops and the induction variable, and nothing else in the
    loop writes memory it reads.
  }];

  let options = [
    Option<"prefetchAcrossIterations", "prefetch-across-iterations", "bool",
           "true",
           "Issue the copy stage of the next iteration of the enclosing loop "
           "during the last compute stage.">,
  ];

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif
