#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHDIALECT
#define MLIR_DIALECT_SNITCH_IR_SNITCHDIALECT

include "mlir/IR/DialectBase.td"

def Snitch_Dialect : Dialect {
  let name = "snitch";
  let cppNamespace = "::mlir::snitch";

  // PipelineOp::verify() calls hasTensorSemantics(), whose default
  // implementation (bufferization::detail::defaultHasTensorSemantics) relies
  // on RankedTensorType implementing the TensorLikeType type interface - an
  // implementation that BufferizationDialect attaches lazily when it loads,
  // not a static property of the builtin type. Without this dependency, a
  // freshly parsed module containing a tensor-semantics `snitch.pipeline` op
  // but no `bufferization.*` ops yet (e.g. before the bufferize pass has run)
  // never loads BufferizationDialect, so the interface is never attached and
  // hasTensorSemantics() silently returns false, tripping PipelineOp's
  // "bufferized pipeline cannot have any iterables" check on a perfectly
  // valid, still-tensor op.
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
  ];

  let discardableAttrs = (ins
    "mlir::StringAttr":$riscv_assembly,
    "mlir::UnitAttr":$xdsl_compilation_failed,
    "mlir::FlatSymbolRefAttr":$dma_specialization
  );

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 0;
  let hasConstantMaterializer = 0;
}

#endif // MLIR_DIALECT_SNITCH_IR_SNITCHDIALECT
