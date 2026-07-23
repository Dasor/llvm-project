
#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTDMATOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateDMAToLLVMConversionPatterns(mlir::ModuleOp moduleOp,
                                         mlir::LLVMTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns);
} // namespace mlir
