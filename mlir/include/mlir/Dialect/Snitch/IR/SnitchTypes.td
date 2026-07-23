#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHTYPES
#define MLIR_DIALECT_SNITCH_IR_SNITCHTYPES

include "mlir/Dialect/Snitch/IR/SnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Snitch_Type<string name, list<Trait> traits = []> :
  TypeDef<Snitch_Dialect, name, traits>;

#endif // MLIR_DIALECT_SNITCH_IR_SNITCHTYPES
