set(LLVM_TARGET_DEFINITIONS SnitchDMADialect.td)
mlir_tablegen(SnitchDMADialect.h.inc -gen-dialect-decls)
mlir_tablegen(SnitchDMADialect.cpp.inc -gen-dialect-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchDMADialectIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchDMAAttrs.td)
mlir_tablegen(SnitchDMAAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(SnitchDMAAttrs.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchDMAAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchDMATypes.td)
mlir_tablegen(SnitchDMATypes.h.inc -gen-typedef-decls)
mlir_tablegen(SnitchDMATypes.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchDMATypesIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchDMAOps.td)
mlir_tablegen(SnitchDMAOps.h.inc -gen-op-decls)
mlir_tablegen(SnitchDMAOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchDMAOpsIncGen)
