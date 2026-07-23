#ifndef MLIR_DIALECT_DMA_IR_DMADIALECT
#define MLIR_DIALECT_DMA_IR_DMADIALECT

include "mlir/IR/DialectBase.td"

def DMA_Dialect : Dialect {
  let name = "dma";
  let cppNamespace = "::mlir::dma";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif // MLIR_DIALECT_DMA_IR_DMADIALECT
