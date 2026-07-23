#ifndef MLIR_DIALECT_DMA_IR_DMAOPS_H
#define MLIR_DIALECT_DMA_IR_DMAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/DMA/IR/DMATypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/DMA/IR/DMAOps.h.inc"

#endif // MLIR_DIALECT_DMA_IR_DMAOPS_H
