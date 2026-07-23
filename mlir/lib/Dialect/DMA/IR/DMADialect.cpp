#include "mlir/Dialect/DMA/IR/DMADialect.h"

#include "mlir/Dialect/DMA/IR/DMADialect.cpp.inc"
#include "mlir/Dialect/DMA/IR/DMAAttrs.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/DMA/IR/DMATypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/DMA/IR/DMAAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::dma;

//===----------------------------------------------------------------------===//
// DMADialect
//===----------------------------------------------------------------------===//

void DMADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DMA/IR/DMAOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/DMA/IR/DMAAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DMA/IR/DMATypes.cpp.inc"
      >();
}

Operation *DMADialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (isa<CompletedTokenAttr>(value))
    return builder.create<CompletedTokenOp>(loc);

  return nullptr;
}
