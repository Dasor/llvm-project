#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DMA/IR/DMADialect.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Returns the 'scf.for' 'computeOp' is in, directly or through 'scf.if's.
static scf::ForOp getEnclosingLoop(Operation *computeOp) {
  Operation *parent = computeOp->getParentOp();
  while (isa_and_nonnull<scf::IfOp>(parent))
    parent = parent->getParentOp();
  return dyn_cast_or_null<scf::ForOp>(parent);
}

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

/// Returns true if 'extractOp', reading from the region iter_arg 'iterArg' of
/// 'forOp', reads a subset that no other iteration writes: the loop yields
/// 'tensor.insert_slice ... into iterArg' at the same subset, and the subset
/// varies with the iteration (as for the output tile of a loop tiling a
/// parallel dimension). The copy of iteration i+1 then reads data that the
/// computation of iteration i does not write.
static bool readsPerIterationSubset(scf::ForOp forOp, BlockArgument iterArg,
                                    tensor::ExtractSliceOp extractOp) {
  if (extractOp.getSource() != iterArg)
    return false;
  auto insertOp = forOp.getTiedLoopYieldedValue(iterArg)
                      ->get()
                      .getDefiningOp<tensor::InsertSliceOp>();
  if (!insertOp || insertOp.getDest() != iterArg)
    return false;
  auto sameValues = [](ArrayRef<OpFoldResult> lhs, ArrayRef<OpFoldResult> rhs) {
    return llvm::equal(lhs, rhs, [](OpFoldResult l, OpFoldResult r) {
      return isEqualConstantIntOrValue(l, r);
    });
  };
  if (!sameValues(extractOp.getMixedOffsets(), insertOp.getMixedOffsets()) ||
      !sameValues(extractOp.getMixedSizes(), insertOp.getMixedSizes()) ||
      !sameValues(extractOp.getMixedStrides(), insertOp.getMixedStrides()))
    return false;
  // A loop-invariant subset is read and written by every iteration.
  return !llvm::all_of(extractOp.getOffsets(), [&](Value offset) {
    return forOp.isDefinedOutsideOfLoop(offset);
  });
}

static void pipelineForLoop(scf::ForOp forOp) {
  // Compute the set of operations that we want to move into the first stage,
  // the copy stage, by calculating the backward slice from the
  // 'wait_for_tensor_copy' operation. The wait operation will remain in the
  // compute stage.
  SetVector<Operation *> toMove;
  // Compute ops directly in the loop body, or in an 'scf.if' in it (as split
  // by 'snitch-split-reduction-remainder').
  SmallVector<TilingInterface> computeOps;
  forOp.getBody()->walk([&](TilingInterface computeOp) {
    if (getEnclosingLoop(computeOp) == forOp)
      computeOps.push_back(computeOp);
  });
  for (TilingInterface computeOp : computeOps) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = true;
    options.filter = [&](Operation *op) {
      return op->getParentRegion() == &forOp.getRegion();
    };

    for (Value operand : computeOp->getOperands()) {
      // TODO: This assumes the immediate operand of the compute op is the
      //       wait operation (or a slice of it, as in a remainder tile split
      //       off by 'snitch-split-reduction-remainder') rather than starting
      //       the backwards slice after the wait operation. Ideally we should
      //       first do one backward slice to discover the wait operations,
      //       then do the backward slice on its transfer tensor.
      while (auto extractOp = operand.getDefiningOp<tensor::ExtractSliceOp>()) {
        if (!forOp->isProperAncestor(extractOp))
          break;
        operand = extractOp.getSource();
      }
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

  // The copy stage of iteration i+1 runs concurrently with the compute stage
  // of iteration i. A copy depending on a loop-carried value (e.g. the output
  // tile of a reduction loop that is copied in and out on every iteration)
  // would read it before the previous iteration produced it. Reading a subset
  // that only the same iteration writes back is fine.
  for (Operation *op : toMove) {
    for (Value operand : op->getOperands()) {
      auto blockArg = dyn_cast<BlockArgument>(operand);
      if (!blockArg || blockArg.getOwner() != forOp.getBody() ||
          blockArg == forOp.getInductionVar())
        continue;
      if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op))
        if (readsPerIterationSubset(forOp, blockArg, extractOp))
          continue;
      InFlightDiagnostic diag = forOp.emitWarning(
          "not pipelining loop: a copy depends on a loop-carried value");
      diag.attachNote(op->getLoc()) << "copy computed here";
      return;
    }
  }

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
    scf::ForOp forOp = getEnclosingLoop(computeOp);
    if (!forOp ||
        !llvm::all_of(forOp.getResultTypes(), llvm::IsaPred<RankedTensorType>))
      return;

    toPipeline.insert(forOp);
  });

  llvm::for_each(toPipeline, pipelineForLoop);
}
