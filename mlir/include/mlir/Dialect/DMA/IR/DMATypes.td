#ifndef MLIR_DIALECT_DMA_IR_DMATYPES
#define MLIR_DIALECT_DMA_IR_DMATYPES

include "mlir/Dialect/DMA/IR/DMADialect.td"
include "mlir/IR/AttrTypeBase.td"

class DMA_Type<string name, list<Trait> traits = []> :
  TypeDef<DMA_Dialect, name, traits>;

def DMA_TokenType : DMA_Type<"Token"> {
  let mnemonic = "token";

  let description = [{
    Type representing a potentially active DMA transfer.
  }];
}

#endif // MLIR_DIALECT_DMA_IR_DMATYPES
