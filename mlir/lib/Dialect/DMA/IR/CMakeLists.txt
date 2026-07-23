add_mlir_dialect_library(MLIRDMADialect
  DMADialect.cpp
  DMAOps.cpp
  DMATypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DMA

  DEPENDS
  MLIRDMADialectIncGen
  MLIRDMAAttrsIncGen
  MLIRDMATypesIncGen
  MLIRDMAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )
