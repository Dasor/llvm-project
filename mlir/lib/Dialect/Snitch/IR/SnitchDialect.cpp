#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"

#include "mlir/Dialect/Snitch/IR/SnitchDialect.cpp.inc"
#include "mlir/Dialect/Snitch/IR/SnitchAttrs.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Dialect/Snitch/IR/SnitchTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Snitch/IR/SnitchAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::snitch;

//===----------------------------------------------------------------------===//
// SnitchDialect
//===----------------------------------------------------------------------===//

void SnitchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Snitch/IR/SnitchOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Snitch/IR/SnitchAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Snitch/IR/SnitchTypes.cpp.inc"
      >();
}
