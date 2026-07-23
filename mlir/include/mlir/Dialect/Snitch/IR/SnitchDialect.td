#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHDIALECT
#define MLIR_DIALECT_SNITCH_IR_SNITCHDIALECT

include "mlir/IR/DialectBase.td"

def Snitch_Dialect : Dialect {
  let name = "snitch";
  let cppNamespace = "::mlir::snitch";

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
