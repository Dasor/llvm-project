#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHATTRS
#define MLIR_DIALECT_SNITCH_IR_SNITCHATTRS

include "mlir/Dialect/Snitch/IR/SnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Snitch_Attr<string name, list<Trait> traits = []> :
  AttrDef<Snitch_Dialect, name, traits>;

def Snitch_L1EncodingAttr : Snitch_Attr<"L1Encoding"> {
  let mnemonic = "l1_encoding";

  let description = [{
    Attribute used as memory space on a memref to denote it being in L1 memory.
  }];
}

#endif // MLIR_DIALECT_SNITCH_IR_SNITCHATTRS
