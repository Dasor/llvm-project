set(LLVM_TARGET_DEFINITIONS SnitchDialect.td)
mlir_tablegen(SnitchDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SnitchDialect.cpp.inc -gen-dialect-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchDialectIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchAttrs.td)
mlir_tablegen(SnitchAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(SnitchAttrs.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchTypes.td)
mlir_tablegen(SnitchTypes.h.inc -gen-typedef-decls)
mlir_tablegen(SnitchTypes.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchTypesIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchInterfaces.td)
mlir_tablegen(SnitchInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SnitchInterfaces.cpp.inc -gen-op-interface-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS SnitchOps.td)
mlir_tablegen(SnitchOps.h.inc -gen-op-decls)
mlir_tablegen(SnitchOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRSnitchOpsIncGen)
