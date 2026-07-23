set(LLVM_TARGET_DEFINITIONS DMADialect.td)
mlir_tablegen(DMADialect.h.inc -gen-dialect-decls)
mlir_tablegen(DMADialect.cpp.inc -gen-dialect-defs)
add_mlir_dialect_tablegen_target(MLIRDMADialectIncGen)

set(LLVM_TARGET_DEFINITIONS DMAAttrs.td)
mlir_tablegen(DMAAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(DMAAttrs.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRDMAAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS DMATypes.td)
mlir_tablegen(DMATypes.h.inc -gen-typedef-decls)
mlir_tablegen(DMATypes.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRDMATypesIncGen)

set(LLVM_TARGET_DEFINITIONS DMAOps.td)
mlir_tablegen(DMAOps.h.inc -gen-op-decls)
mlir_tablegen(DMAOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRDMAOpsIncGen)
