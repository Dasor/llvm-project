
#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTSNITCHTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateSnitchToLLVMConversionPatterns(mlir::ModuleOp moduleOp,
                                            mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns);
} // namespace mlir
