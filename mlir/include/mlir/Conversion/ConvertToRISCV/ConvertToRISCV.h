
#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTTORISCVPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir
