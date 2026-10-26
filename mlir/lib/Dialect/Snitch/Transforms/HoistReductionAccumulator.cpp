#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_HOISTREDUCTIONACCUMULATORPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class HoistReductionAccumulator
    : public mlir::snitch::impl::HoistReductionAccumulatorPassBase<
          HoistReductionAccumulator> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;

/// Returns true if all of 'values' are defined outside of 'loop'.
static bool allDefinedOutside(scf::ForOp loop, ValueRange values) {
  return llvm::all_of(
      values, [&](Value value) { return loop.isDefinedOutsideOfLoop(value); });
}

/// Rewrites a loop-carried accumulator of 'computeOp' so that the loop carries
/// exactly the tensor 'computeOp' accumulates into, instead of re-deriving it
/// from the iter_arg every iteration. Matches (after subset hoisting):
///
///   scf.for ... iter_args(%acc = %init) {
///     [%p = tensor.pad %acc low[0, ...] high[<invariant>] { yield 0 }]
///     %r = <computeOp> ... outs(%p or %acc)
///     [%u = tensor.extract_slice %r[0, ...] [<invariant>] [1, ...]]
///     [%m = bufferization.materialize_in_destination %u or %r in %acc]
///     scf.yield %m or %u or %r
///   }
///
/// and rewrites it to carry %r directly: the pad (if any) is applied to %init
/// before the loop and the unpadding 'extract_slice' (if any) to the loop
/// result after it. Does nothing if the pattern does not match exactly.
static void carryAccumulator(RewriterBase &rewriter, scf::ForOp loop,
                             DestinationStyleOpInterface computeOp) {
  for (OpOperand &initOperand : computeOp.getDpsInitsMutable()) {
    Value init = initOperand.get();
    auto padOp = init.getDefiningOp<tensor::PadOp>();
    Value accValue = padOp ? padOp.getSource() : init;

    auto acc = dyn_cast<BlockArgument>(accValue);
    if (!acc || acc.getOwner() != loop.getBody() ||
        acc == loop.getInductionVar())
      continue;

    if (padOp) {
      if (!padOp->hasOneUse() || !padOp.hasZeroLowPad() ||
          !allDefinedOutside(loop, padOp.getHigh()))
        continue;
      Value padValue = padOp.getConstantPaddingValue();
      if (!padValue || !matchPattern(padValue, m_AnyZeroFloat()))
        continue;
    }

    // Peel the yielded value back to the compute op's result.
    OpOperand *yielded = loop.getTiedLoopYieldedValue(acc);
    Value value = yielded->get();
    auto materializeOp =
        value.getDefiningOp<bufferization::MaterializeInDestinationOp>();
    if (materializeOp) {
      if (materializeOp.getDest() != acc || !materializeOp->hasOneUse())
        continue;
      value = materializeOp.getSource();
    }
    tensor::ExtractSliceOp unpadOp;
    if (padOp) {
      unpadOp = value.getDefiningOp<tensor::ExtractSliceOp>();
      if (!unpadOp || !unpadOp->hasOneUse() || !unpadOp.hasZeroOffset() ||
          !unpadOp.hasUnitStride() ||
          !allDefinedOutside(loop, unpadOp.getSizes()) ||
          unpadOp.getType() != acc.getType())
        continue;
      value = unpadOp.getSource();
    }
    OpResult result = computeOp.getTiedOpResult(&initOperand);
    if (value != result)
      continue;

    if (!padOp && !materializeOp)
      continue; // Already in the desired form.
    // The accumulator must not be used by anything else in the loop: only by
    // the pad (or the compute op) and the 'materialize_in_destination'.
    unsigned expectedUses = materializeOp ? 2 : 1;
    if (std::distance(acc.use_begin(), acc.use_end()) != expectedUses)
      continue;

    // Rewrite. Pad the init before the loop.
    OpOperand *loopInit = loop.getTiedLoopInit(acc);
    OpResult loopResult = loop.getTiedLoopResult(acc);
    if (padOp) {
      rewriter.setInsertionPoint(loop);
      IRMapping mapping;
      mapping.map(acc, loopInit->get());
      Operation *newPad = rewriter.clone(*padOp, mapping);
      rewriter.modifyOpInPlace(
          loop, [&] { loopInit->set(newPad->getResult(0)); });
      acc.setType(padOp.getType());
      loopResult.setType(padOp.getType());
      rewriter.replaceAllUsesWith(padOp.getResult(), acc);
    }
    rewriter.modifyOpInPlace(yielded->getOwner(),
                             [&] { yielded->set(result); });
    if (materializeOp)
      rewriter.eraseOp(materializeOp);
    if (padOp) {
      // Unpad after the loop.
      rewriter.setInsertionPointAfter(loop);
      auto newUnpad = rewriter.create<tensor::ExtractSliceOp>(
          unpadOp.getLoc(), unpadOp.getType(), loopResult,
          unpadOp.getMixedOffsets(), unpadOp.getMixedSizes(),
          unpadOp.getMixedStrides());
      rewriter.replaceAllUsesExcept(loopResult, newUnpad.getResult(),
                                    newUnpad);
      rewriter.eraseOp(unpadOp);
      rewriter.eraseOp(padOp);
    }
  }
}

void HoistReductionAccumulator::runOnOperation() {
  SmallVector<linalg::LinalgOp> computeOps;
  getOperation()->walk([&](linalg::LinalgOp computeOp) {
    if (isa<scf::ForOp>(computeOp->getParentOp()) &&
        computeOp.hasPureTensorSemantics())
      computeOps.push_back(computeOp);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::LinalgOp computeOp : computeOps) {
    auto loop = cast<scf::ForOp>(computeOp->getParentOp());
    // Move loop-invariant sizes/offsets of the accumulator's subset out first,
    // which subset hoisting requires.
    moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
    // Hoists the 'extract_slice'/'insert_slice' pair of the accumulator. This
    // replaces the loop with one carrying the slice as an additional iter_arg.
    hoistLoopInvariantSubsets(rewriter,
                              cast<LoopLikeOpInterface>(loop.getOperation()));
    loop = cast<scf::ForOp>(computeOp->getParentOp());
    carryAccumulator(rewriter, loop,
                     cast<DestinationStyleOpInterface>(computeOp.getOperation()));
  }
}
