add_mlir_dialect_library(MLIRSnitchDMADialect
  SnitchDMADialect.cpp
  SnitchDMAOps.cpp
  SnitchDMATypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SnitchDMA

  DEPENDS
  MLIRSnitchDMADialectIncGen
  MLIRSnitchDMAAttrsIncGen
  MLIRSnitchDMATypesIncGen
  MLIRSnitchDMAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  )
