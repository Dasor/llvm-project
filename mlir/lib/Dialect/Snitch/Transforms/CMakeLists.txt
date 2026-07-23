add_mlir_dialect_library(MLIRSnitchTransforms
  FormMicrokernels.cpp
  LowerL1Allocations.cpp
  PromoteToL1.cpp
  SpecializeDMACode.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Snitch/Transforms

  DEPENDS
  MLIRSnitchTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDMADialect
  MLIRSnitchDialect
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFunctionInterfaces
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRUBDialect
  )
