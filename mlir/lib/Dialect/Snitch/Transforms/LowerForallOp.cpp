#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_LOWERFORALLOPPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class LowerForallOp
    : public mlir::snitch::impl::LowerForallOpPassBase<LowerForallOp> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace mlir::snitch;

void LowerForallOp::runOnOperation() {
  getOperation()->walk([&](scf::ForallOp forallOp) {
    // Only single-induction-variable foralls are supported - this is what
    // `transform.structured.tile_using_forall ... num_threads [N, 0, 0]`
    // (with all but one dimension's num_threads set to a static 0) produces.
    if (forallOp.getInductionVars().size() != 1)
      return;

    OpBuilder builder(forallOp);
    Location loc = forallOp.getLoc();

    Value lb = forallOp.getLowerBound(builder).front();
    Value ub = forallOp.getUpperBound(builder).front();
    Value step = forallOp.getStep(builder).front();

    Value id = builder.create<ComputeCoreIndexOp>(loc);

    // Hart `id` processes iterations `lb + id*step, lb + id*step +
    // computeCores*step, ...` - a strided, round-robin static distribution
    // across `computeCores` harts (all of which execute this same code,
    // differentiated only by ComputeCoreIndexOp's runtime value).
    Value newLb = affine::makeComposedAffineApply(
        builder, loc,
        builder.getAffineDimExpr(0) +
            builder.getAffineDimExpr(1) * builder.getAffineDimExpr(2),
        {lb, id, step});
    Value newStep = affine::makeComposedAffineApply(
        builder, loc,
        builder.getAffineConstantExpr(computeCores) *
            builder.getAffineDimExpr(0),
        {step});

    forallOp.getTerminator().erase();

    auto forOp = builder.create<scf::ForOp>(loc, newLb, ub, newStep);
    forOp.getRegion().takeBody(forallOp.getRegion());
    builder.setInsertionPointToEnd(forOp.getBody());
    builder.create<scf::YieldOp>(loc);

    forallOp.erase();
  });
}
