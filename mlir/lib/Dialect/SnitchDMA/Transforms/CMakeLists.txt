add_mlir_dialect_library(MLIRSnitchDMATransforms
  LegalizeDMAOperations.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SnitchDMA/Transforms

  DEPENDS
  MLIRSnitchDMATransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDMADialect
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  )
