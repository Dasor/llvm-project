
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/SnitchDMA/IR/SnitchDMATypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.h.inc"

namespace mlir::snitch_dma {
class QueueResource : public mlir::SideEffects::Resource::Base<QueueResource> {
public:
  llvm::StringRef getName() const override;
};
} // namespace mlir::snitch_dma
