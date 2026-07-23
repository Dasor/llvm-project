#ifndef MLIR_DIALECT_SNITCHDMA_TRANSFORMS_PASSES
#define MLIR_DIALECT_SNITCHDMA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LegalizeDMAOperationsPass : Pass<"snitch-dma-legalize-dma-operations"> {
  let description = [{

  }];

  let dependentDialects = [
    "mlir::dma::DMADialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif
