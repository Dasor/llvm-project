#include "mlir/Dialect/SnitchDMA/IR/SnitchDMADialect.h"

#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAAttrs.h"
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.h"
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMATypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAAttrs.cpp.inc"

#include "mlir/Dialect/SnitchDMA/IR/SnitchDMADialect.cpp.inc"

using namespace mlir;
using namespace mlir::snitch_dma;

//===----------------------------------------------------------------------===//
// SnitchDMADialect
//===----------------------------------------------------------------------===//

void SnitchDMADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMATypes.cpp.inc"
      >();
}
