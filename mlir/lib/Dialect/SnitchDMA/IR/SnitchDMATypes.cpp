#include "mlir/Dialect/SnitchDMA/IR/SnitchDMATypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/SnitchDMA/IR/SnitchDMADialect.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMATypes.cpp.inc"
