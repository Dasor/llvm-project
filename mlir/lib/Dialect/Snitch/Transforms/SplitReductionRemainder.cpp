#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_SPLITREDUCTIONREMAINDERPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class SplitReductionRemainder
    : public mlir::snitch::impl::SplitReductionRemainderPassBase<
          SplitReductionRemainder> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;

/// Splits the last, partial iteration off the compute op of a reduction loop
/// whose step doesn't divide its trip range. Matches
///
///   scf.for %iv = <lb> to <ub> step <k> iter_args(%acc = ...) {
///     %a = tensor.pad %a_slice ... { yield %zero }   // reduction dim -> k
///     %b = tensor.pad %b_slice ... { yield %zero }
///     %r = <computeOp> ins(%a, %b) outs(%acc)       // one reduction dim, of size k
///     ...
///   }
///
/// with constant bounds, where (ub - lb) % k = rem != 0, and rewrites it to
///
///     %a = tensor.pad %a_slice ... { yield poison }
///     %b = tensor.pad %b_slice ... { yield poison }
///     %r = scf.if %iv != <start of the last iteration> {
///       <computeOp> ins(%a, %b) outs(%acc)
///     } else {
///       <computeOp> ins(%a[.., 0:rem, ..], %b[.., 0:rem, ..]) outs(%acc)
///     }
///
/// The padded tiles keep their uniform shape, so the loop can still be
/// dual-buffered with the same buffers in every iteration, but the last
/// iteration only computes on the valid part of its tiles, and nothing is
/// zero-filled: once the padding is never read, it can be left undefined.
static void splitRemainder(RewriterBase &rewriter, linalg::LinalgOp computeOp) {
  auto loop = dyn_cast<scf::ForOp>(computeOp->getParentOp());
  if (!loop || !computeOp.hasPureTensorSemantics() ||
      computeOp->getNumRegions() != 1)
    return;
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return;
  int64_t rem = (*ub - *lb) % *step;
  if (rem == 0)
    return;

  // Exactly one reduction dimension, tiled by the loop to its step.
  SmallVector<unsigned> reductionDims;
  computeOp.getReductionDims(reductionDims);
  if (reductionDims.size() != 1)
    return;
  unsigned dim = reductionDims.front();
  SmallVector<int64_t> ranges = computeOp.getStaticLoopRanges();
  if (ranges[dim] != *step)
    return;

  // Every input indexed by the reduction dimension must be a zero low-padded
  // tile padded with a constant: the padding the remainder no longer reads.
  struct Input {
    OpOperand *operand;
    tensor::PadOp padOp;
    SmallVector<int64_t> sizes; // of the remainder slice
  };
  SmallVector<Input> inputs;
  for (OpOperand *operand : computeOp.getDpsInputOperands()) {
    AffineMap map = computeOp.getMatchingIndexingMap(operand);
    auto type = dyn_cast<RankedTensorType>(operand->get().getType());
    if (!type || !type.hasStaticShape())
      return;
    SmallVector<int64_t> sizes = llvm::to_vector(type.getShape());
    bool indexed = false;
    for (auto [index, expr] : llvm::enumerate(map.getResults())) {
      if (!expr.isFunctionOfDim(dim))
        continue;
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return;
      sizes[index] = rem;
      indexed = true;
    }
    if (!indexed)
      continue;
    auto padOp = operand->get().getDefiningOp<tensor::PadOp>();
    if (!padOp || !padOp->hasOneUse() || !padOp.hasZeroLowPad() ||
        padOp->getParentOp() != loop || !padOp.getConstantPaddingValue())
      return;
    inputs.push_back({operand, padOp, sizes});
  }
  if (inputs.empty())
    return;

  // The padding is never read any more: leave it undefined.
  rewriter.setInsertionPoint(loop);
  DenseMap<Type, Value> poisons;
  for (Input &input : inputs) {
    Type elementType = input.padOp.getResultType().getElementType();
    Value &poison = poisons[elementType];
    if (!poison)
      poison = ub::PoisonOp::create(rewriter, input.padOp.getLoc(), elementType);
    auto yieldOp = cast<tensor::YieldOp>(
        input.padOp.getRegion().front().getTerminator());
    rewriter.modifyOpInPlace(yieldOp,
                             [&] { yieldOp.getValueMutable().assign(poison); });
  }
  Location loc = computeOp.getLoc();
  Value last = arith::ConstantIndexOp::create(rewriter, loc, *ub - rem);

  rewriter.setInsertionPoint(computeOp);
  Value notLast = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        loop.getInductionVar(), last);
  auto ifOp = scf::IfOp::create(rewriter, loc, computeOp->getResultTypes(),
                                notLast, /*withElseRegion=*/true);

  // Full tile.
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  Operation *full = rewriter.clone(*computeOp);
  scf::YieldOp::create(rewriter, loc, full->getResults());

  // Remainder: the same op on the valid part of the tiles.
  rewriter.setInsertionPointToStart(ifOp.elseBlock());
  SmallVector<Value> operands = computeOp->getOperands();
  for (Input &input : inputs) {
    auto type = cast<RankedTensorType>(input.operand->get().getType());
    unsigned rank = type.getRank();
    auto slice = tensor::ExtractSliceOp::create(
        rewriter, loc,
        RankedTensorType::get(input.sizes, type.getElementType()),
        input.operand->get(), SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0)),
        getAsIndexOpFoldResult(rewriter.getContext(), input.sizes),
        SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));
    operands[input.operand->getOperandNumber()] = slice;
  }
  Operation *remainder =
      clone(rewriter, computeOp, computeOp->getResultTypes(), operands);
  // Discardable attributes (e.g. a dual-buffering marker) stay with the full
  // tile, whose loop they describe.
  for (NamedAttribute attr : computeOp->getDiscardableAttrs())
    remainder->removeDiscardableAttr(attr.getName());
  scf::YieldOp::create(rewriter, loc, remainder->getResults());

  rewriter.replaceOp(computeOp, ifOp.getResults());
}

void SplitReductionRemainder::runOnOperation() {
  SmallVector<linalg::LinalgOp> computeOps;
  getOperation()->walk(
      [&](linalg::LinalgOp computeOp) { computeOps.push_back(computeOp); });

  IRRewriter rewriter(&getContext());
  for (linalg::LinalgOp computeOp : computeOps)
    splitRemainder(rewriter, computeOp);
}
