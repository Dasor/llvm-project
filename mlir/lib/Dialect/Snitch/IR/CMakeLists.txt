add_mlir_dialect_library(MLIRSnitchDialect
  SnitchDialect.cpp
  SnitchInterfaces.cpp
  SnitchOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Snitch

  DEPENDS
  MLIRSnitchDialectIncGen
  MLIRSnitchAttrsIncGen
  MLIRSnitchTypesIncGen
  MLIRSnitchInterfacesIncGen
  MLIRSnitchOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )
