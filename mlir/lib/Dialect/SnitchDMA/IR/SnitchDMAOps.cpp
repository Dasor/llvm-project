#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.cpp.inc"

using namespace mlir;
using namespace mlir::snitch_dma;

StringRef QueueResource::getName() const {
  return "queue";
}
