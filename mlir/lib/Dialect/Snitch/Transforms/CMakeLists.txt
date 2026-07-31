add_mlir_dialect_library(MLIRSnitchTransforms
  Bufferize.cpp
  FormMicrokernels.cpp
  LowerForallOp.cpp
  LowerL1Allocations.cpp
  LowerPipelineOp.cpp
  PipelineCopyCompute.cpp
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
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFunctionInterfaces
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRUBDialect
  )
