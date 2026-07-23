add_mlir_conversion_library(MLIRConvertToRISCV
  ConvertToRISCV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ConvertToRISCV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSnitchDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  )
