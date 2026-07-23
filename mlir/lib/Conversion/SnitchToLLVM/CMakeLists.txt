add_mlir_conversion_library(MLIRSnitchToLLVM
  SnitchToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SnitchToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSnitchDialect
  MLIRAnalysis
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransforms
  )
