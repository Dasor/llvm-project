#ifndef MLIR_DIALECT_SNITCH_TRANSFORMS_PASSES
#define MLIR_DIALECT_SNITCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormMicrokernelsPass
  : InterfacePass<"snitch-form-microkernels", "mlir::FunctionOpInterface"> {
  let dependentDialects = [
    "mlir::snitch::SnitchDialect",
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

#endif
