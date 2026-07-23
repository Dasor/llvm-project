#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHOPS_H
#define MLIR_DIALECT_SNITCH_IR_SNITCHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Snitch/IR/SnitchInterfaces.h"
#include "mlir/Dialect/Snitch/IR/SnitchTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Snitch/IR/SnitchOps.h.inc"

#endif // MLIR_DIALECT_SNITCH_IR_SNITCHOPS_H
