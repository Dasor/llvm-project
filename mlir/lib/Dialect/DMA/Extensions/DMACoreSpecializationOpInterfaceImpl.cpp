#include "mlir/Dialect/DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.h"

#include "mlir/Dialect/DMA/IR/DMADialect.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/Snitch/IR/SnitchInterfaces.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::dma;
using namespace mlir::snitch;

namespace {

//===----------------------------------------------------------------------===//
// StartTransferOp::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct StartTransferOpImpl
    : CoreSpecializationOpInterface::ExternalModel<StartTransferOpImpl,
                                                   StartTransferOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
  }
};

struct StartTransferOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<StartTransferOpDMAImpl,
                                                      StartTransferOp> {};

//===----------------------------------------------------------------------===//
// StartZeroMemTransferOp::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct StartZeroMemTransferOpImpl
    : CoreSpecializationOpInterface::ExternalModel<StartZeroMemTransferOpImpl,
                                                   StartZeroMemTransferOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
  }

  // bool needsSynchronization(Operation *op) const { return true; }
};

struct StartZeroMemTransferOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<
          StartZeroMemTransferOpDMAImpl, StartZeroMemTransferOp> {};

//===----------------------------------------------------------------------===//
// WaitForTransferOpImpl::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct WaitForTransferOpImpl
    : CoreSpecializationOpInterface::ExternalModel<WaitForTransferOpImpl,
                                                   WaitForTransferOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.eraseOp(op);
  }

  bool needsSynchronization(Operation *op) const { return true; }
};

struct WaitForTransferOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<WaitForTransferOpDMAImpl,
                                                      WaitForTransferOp> {};

} // namespace

void mlir::dma::registerDMACoreSpecializationOpInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, DMADialect *dialect) {
#define REGISTER_IMPLS(Op) Op::attachInterface<Op##Impl, Op##DMAImpl>(*context)
    REGISTER_IMPLS(StartTransferOp);
    REGISTER_IMPLS(StartZeroMemTransferOp);
    REGISTER_IMPLS(WaitForTransferOp);
  });
}
