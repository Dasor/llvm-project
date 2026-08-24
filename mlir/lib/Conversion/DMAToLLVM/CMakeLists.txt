add_mlir_conversion_library(MLIRDMAToLLVM
  DMAToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/DMAToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRDMADialect
  MLIRSnitchDMADialect
  MLIRAnalysis
  MLIRControlFlowDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransforms
  )
