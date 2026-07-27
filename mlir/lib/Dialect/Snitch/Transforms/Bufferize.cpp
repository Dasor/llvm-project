#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/DMA/IR/DMADialect.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_ELIMINATEEMPTYTENSORSPASS
#define GEN_PASS_DEF_BUFFERIZEPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

using namespace mlir;
using namespace mlir::bufferization;

namespace {

/// Builds a filter matching upstream's own bufferization pass: constants get
/// their own dedicated one-shot bufferize invocation afterward (so they can
/// use a different set of options), and `to_buffer` ops are boundary
/// materializations that must not be bufferized themselves.
static OneShotBufferizationOptions getBufferizationOptions() {
  OneShotBufferizationOptions options;
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToBufferOp>();
  return options;
}

struct EliminateEmptyTensorsPass
    : public mlir::snitch::impl::EliminateEmptyTensorsPassBase<
          EliminateEmptyTensorsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      linalg::populateConvertToDestinationStylePatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
        return signalPassFailure();
    }

    IRRewriter rewriter(context);
    OneShotBufferizationOptions options = getBufferizationOptions();
    OneShotAnalysisState state(funcOp, options);
    if (failed(analyzeOp(funcOp, state)))
      return signalPassFailure();
    if (failed(bufferization::eliminateEmptyTensors(rewriter, funcOp, state)))
      return signalPassFailure();
  }
};

struct BufferizePass
    : public mlir::snitch::impl::BufferizePassBase<BufferizePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    OneShotBufferizationOptions options = getBufferizationOptions();
    // Turning this off assumes we are not relying on bufferization being
    // conservative around parallel regions (e.g. `scf.forall`). Revisit if
    // dual-buffered pipelining (Milestone 4) exposes a data race here.
    options.checkParallelRegions = false;
    // Unlike IREE (which converts dispatch region signatures to memrefs via
    // its own HAL ABI before this point), we bufferize function signatures
    // here directly. Use a static identity layout so exported functions keep
    // a C-ABI-compatible memref type, required by xDSL's bare-pointer calling
    // convention for microkernel arguments. This requires the module-level
    // (call-graph-aware) bufferization entry point below rather than
    // bufferizing each function in isolation.
    options.bufferizeFunctionBoundaries = true;
    options.setFunctionBoundaryTypeConversion(LayoutMapOption::IdentityLayoutMap);

    if (useDMAMemcpy) {
      options.allocationFn = [](OpBuilder &builder, Location loc,
                                MemRefType memRefType, ValueRange dynamicSizes,
                                unsigned alignment) -> Value {
        return builder.create<memref::AllocaOp>(
            loc, memRefType, dynamicSizes, builder.getI64IntegerAttr(alignment));
      };
      options.memCpyFn = [](OpBuilder &builder, Location loc, Value from,
                            Value to) {
        Value token = builder.create<dma::StartTransferOp>(loc, from, to);
        builder.create<dma::WaitForTransferOp>(loc, token);
        return success();
      };
    }

    BufferizationState state;
    if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options, state)))
      return signalPassFailure();

    RewritePatternSet patterns(&getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
