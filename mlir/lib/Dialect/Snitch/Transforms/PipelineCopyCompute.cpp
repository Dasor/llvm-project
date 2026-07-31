#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DMA/IR/DMADialect.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_PIPELINECOPYCOMPUTEPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class PipelineCopyCompute
    : public mlir::snitch::impl::PipelineCopyComputePassBase<
          PipelineCopyCompute> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

} // namespace

using namespace mlir;
using namespace mlir::snitch;
using namespace mlir::dma;

/// Unit attribute attached (e.g. via `transform.annotate`) to the compute
/// op inside a tiling loop that should be dual-buffered/pipelined. This is
/// the non-IREE replacement for the original `LoweringConfigAttr`'s
/// `getDualBuffer()` field - a plain marker rather than a lowering-config
/// lookup, since this prototype has no IREE codegen lowering-config
/// infrastructure.
static constexpr llvm::StringLiteral kDualBufferAttrName = "snitch.dual_buffer";

/// Lifts an 'scf.for' op to a pipeline op with two stages.
/// The body of the for loop gets placed in the second stage with all iter args
/// forwarded from the first to the second stage.
static PipelineOp liftToPipeline(scf::ForOp forOp) {
  OpBuilder builder(forOp);
  auto toIndex = [&](Value value) -> Value {
    if (isa<IndexType>(value.getType()))
      return value;
    return builder.create<arith::IndexCastOp>(value.getLoc(),
                                              builder.getIndexType(), value);
  };
  auto pipelineOp = builder.create<PipelineOp>(
      forOp.getLoc(), forOp->getResultTypes(), toIndex(forOp.getLowerBound()),
      toIndex(forOp.getUpperBound()), toIndex(forOp.getStep()),
      forOp.getInitArgs(),
      /*numRegions=*/2);

  Region &copyRegion = pipelineOp.getStages()[0];
  Region &computeRegion = pipelineOp.getStages()[1];
  computeRegion.takeBody(forOp.getRegion());
  {
    OpBuilder::InsertionGuard guard{builder};
    Block &singleBlock = computeRegion.front();
    builder.setInsertionPointToEnd(&singleBlock);
    auto yieldOp = cast<scf::YieldOp>(singleBlock.getTerminator());
    builder.create<PipelineYieldOp>(yieldOp.getLoc(), yieldOp.getResults());
    yieldOp->erase();
  }
  {
    OpBuilder::InsertionGuard guard{builder};

    builder.setInsertionPointToStart(&copyRegion.emplaceBlock());
    copyRegion.addArguments(computeRegion.getArgumentTypes(),
                            llvm::map_to_vector(computeRegion.getArguments(),
                                                std::mem_fn(&Value::getLoc)));
    // Don't yield the induction variable.
    builder.create<PipelineYieldOp>(pipelineOp.getLoc(),
                                    copyRegion.getArguments().drop_front());
  }
  return pipelineOp;
}

static void pipelineForLoop(scf::ForOp forOp) {
  // Compute the set of operations that we want to move into the first stage,
  // the copy stage, by calculating the backward slice from the
  // 'wait_for_tensor_copy' operation. The wait operation will remain in the
  // compute stage.
  SetVector<Operation *> toMove;
  for (auto computeOp : forOp.getOps<TilingInterface>()) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = true;
    options.filter = [&](Operation *op) {
      return op->getParentRegion() == &forOp.getRegion();
    };

    for (Value operand : computeOp->getOperands()) {
      // TODO: This assumes the immediate operand of the compute op is the
      //       wait operation rather than starting the backwards slice after the
      //       wait operation. Ideally we should first do one backward slice to
      //       discover the wait operations, then do the backward slice on its
      //       transfer tensor.
      auto tensorCopyWait = operand.getDefiningOp<WaitForTensorCopyOp>();
      if (!tensorCopyWait)
        continue;

      // Ignore if outside the for loop.
      if (tensorCopyWait->getParentRegion() != &forOp.getRegion())
        continue;

      if (isa<BlockArgument>(tensorCopyWait.getTransferTensor()))
        continue;

      getBackwardSlice(tensorCopyWait.getTransferTensor(), &toMove, options);
    }
  }
  if (!llvm::all_of(toMove, isPure))
    return;

  PipelineOp pipelineOp = liftToPipeline(forOp);
  forOp.replaceAllUsesWith(pipelineOp.getResults());
  forOp->erase();

  // Move the copy operations into the copy region.
  Region &copyRegion = pipelineOp.getStages()[0];
  Region &computeRegion = pipelineOp.getStages()[1];
  auto copyRegionBuilder = OpBuilder::atBlockBegin(&copyRegion.front());
  for (Operation *op : toMove) {
    op->remove();
    copyRegionBuilder.insert(op);
    // Make sure to remap the induction variable from the one of the compute
    // region to the copy region.
    for (OpOperand &operand : op->getOpOperands())
      if (auto blockArg = dyn_cast<BlockArgument>(operand.get()))
        if (blockArg.getParentRegion() == &computeRegion)
          operand.set(copyRegion.getArgument(blockArg.getArgNumber()));
  }

  // Fix up any uses in the compute region that are defined in the copy region.
  // On the first occurrence of such a use, we yield the value and rewrite the
  // use to a new corresponding block argument in the compute region.
  PipelineYieldOp copyRegionYield =
      cast<PipelineYieldOp>(pipelineOp.getStages()[0].front().getTerminator());
  for (Operation *op : toMove) {
    for (OpResult result : op->getResults()) {
      BlockArgument replacement;
      for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
        // Skip uses in the copy region.
        if (copyRegion.isAncestor(use.getOwner()->getParentRegion()))
          continue;

        if (!replacement) {
          replacement =
              computeRegion.addArgument(result.getType(), op->getLoc());
          copyRegionYield.getResultsMutable().append(result);
        }
        use.set(replacement);
      }
    }
  }
}

void PipelineCopyCompute::runOnOperation() {
  // Collect surrounding for loops of ops that requested to be multi buffered.
  SetVector<scf::ForOp> toPipeline;
  getOperation()->walk([&](TilingInterface computeOp) {
    if (!computeOp->hasAttr(kDualBufferAttrName))
      return;

    // TODO: This creates an uncomfortably tight coupling between the
    //  pass pipeline and tiling and the dual buffering.
    auto forOp = dyn_cast<scf::ForOp>(computeOp->getParentOp());
    if (!forOp ||
        !llvm::all_of(forOp.getResultTypes(), llvm::IsaPred<RankedTensorType>))
      return;

    toPipeline.insert(forOp);
  });

  llvm::for_each(toPipeline, pipelineForLoop);
}
