#include "mlir/Dialect/DMA/IR/DMATypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/DMA/IR/DMADialect.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DMA/IR/DMATypes.cpp.inc"
