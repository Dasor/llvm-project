#ifndef MLIR_DIALECT_SNITCHDMA_IR_SNITCHDMAATTRS
#define MLIR_DIALECT_SNITCHDMA_IR_SNITCHDMAATTRS

include "mlir/Dialect/SnitchDMA/IR/SnitchDMADialect.td"
include "mlir/IR/AttrTypeBase.td"

class SnitchDMA_Attr<string name, list<Trait> traits = []> :
  AttrDef<SnitchDMA_Dialect, name, traits>;


#endif
