
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::snitch_dma {
#define GEN_PASS_DECL
#include "mlir/Dialect/SnitchDMA/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SnitchDMA/Transforms/Passes.h.inc"
} // namespace mlir::snitch_dma
