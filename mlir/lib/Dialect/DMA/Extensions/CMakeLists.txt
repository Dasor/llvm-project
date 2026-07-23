add_mlir_extension_library(MLIRDMACoreSpecializationOpInterfaceImpl
  DMACoreSpecializationOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DMA/Extensions

  LINK_LIBS PUBLIC
  MLIRDMADialect
  MLIRSnitchDialect
  MLIRIR
  )
