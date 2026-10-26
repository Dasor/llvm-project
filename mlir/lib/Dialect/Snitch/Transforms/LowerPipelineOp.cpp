#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DMA/IR/DMAOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Dialect/Snitch/IR/SnitchInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "snitch-lower-pipeline-op"

namespace mlir::snitch {
#define GEN_PASS_DEF_LOWERPIPELINEOPPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class LowerPipelineOp
    : public mlir::snitch::impl::LowerPipelineOpPassBase<LowerPipelineOp> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace mlir::snitch;

/// Duplicate resources required by a stage to support concurrent execution of
/// the stage and switches on it. Currently only supports `memref.alloca`.
/// Returns the duplicated allocations, which are inserted right before the
/// pipeline op.
static SmallVector<memref::AllocaOp>
makeResourceUsageExplicit(PipelineOp pipelineOp) {
  SmallVector<memref::AllocaOp> resources;
  pipelineOp->walk([&](memref::AllocaOp allocation) {
    // TODO: Pipeline independent dynamic allocations would be fine.
    assert(allocation.getDynamicSizes().empty() &&
           "dynamic allocations not supported");
    Location loc = allocation->getLoc();

    // Clone right before the pipeline op as many times as we have stages.
    SmallVector<memref::AllocaOp> replacements;
    OpBuilder builder(pipelineOp);
    for ([[maybe_unused]] unsigned i : llvm::seq(pipelineOp.getStages().size()))
      builder.insert(replacements.emplace_back(allocation.clone()));

    // Find the region the allocation is contained in to find the IV we need to
    // use for the resource cycling.
    Region *parentRegion = allocation->getParentRegion();
    while (parentRegion->getParentOp() != pipelineOp)
      parentRegion = parentRegion->getParentRegion();
    Value iv = parentRegion->getArgument(0);

    // The cycle can be computed by normalizing the IV first and then
    // performing a mod operation using our number of stages.
    builder.setInsertionPoint(allocation);
    Value cycle = affine::makeComposedAffineApply(
        builder, loc,
        builder.getAffineDimExpr(0).floorDiv(builder.getAffineDimExpr(1)) %
            pipelineOp.getStages().size(),
        {iv, pipelineOp.getStep()});

    // Switch on the cycle number and return the corresponding resource.
    auto switchOp = builder.create<scf::IndexSwitchOp>(
        loc, allocation.getType(), cycle,
        llvm::to_vector(llvm::seq<int64_t>(pipelineOp.getStages().size() - 1)),
        pipelineOp.getStages().size() - 1);
    for (unsigned i : llvm::seq(pipelineOp.getRegions().size())) {
      if (i == pipelineOp.getRegions().size() - 1)
        builder.setInsertionPointToStart(
            &switchOp.getDefaultRegion().emplaceBlock());
      else
        builder.setInsertionPointToStart(
            &switchOp.getCaseRegions()[i].emplaceBlock());

      builder.create<scf::YieldOp>(loc, replacements[i].getMemref());
    }

    allocation->replaceAllUsesWith(switchOp);
    allocation->erase();
    llvm::append_range(resources, replacements);
  });
  return resources;
}

/// Clones the given pipeline stage at the insertion point
/// of builder without the yield operation.
/// 'mapping' can be used to remap the block arguments of the block
/// and will additionally be used to contain the result mapping.
///
/// Returns the operands of the yield operation had it been cloned.
/// The first 'skip' operations of the stage are not cloned.
static SmallVector<Value> insertStage(OpBuilder &builder, Region &stage,
                                      IRMapping &mapping, unsigned skip = 0) {
  Block *block = &stage.front();
  for (Operation &op : llvm::drop_begin(block->without_terminator(), skip))
    builder.clone(op, mapping);

  SmallVector<Value> result;
  for (Value value : cast<PipelineYieldOp>(block->getTerminator()).getResults())
    result.push_back(mapping.lookupOrDefault(value));

  return result;
}

/// Returns true if 'op' waits for a DMA transfer started outside of 'stage',
/// i.e. by an earlier stage of the same iteration.
static bool isWaitForEarlierStage(Operation &op, Region &stage) {
  return isa<dma::WaitForTransferOp>(op) &&
         llvm::all_of(op.getOperands(), [&](Value operand) {
           return !stage.isAncestor(operand.getParentRegion()) ||
                  isa<BlockArgument>(operand);
         });
}

/// Inserts 'stages' (with the block argument mappings 'mappings'), which run
/// concurrently in one step of the pipeline, i.e. for different iterations.
/// Returns the would-be yielded values of each stage.
///
/// The waits a later stage starts with (for transfers issued by an earlier
/// stage in a previous step) are inserted before all stages: they do not
/// depend on the earlier stages of this step. On the DMA core, this lets the
/// compute cores be released as soon as their data has arrived, rather than
/// after the DMA core has issued the transfers of the next iteration.
static SmallVector<SmallVector<Value>>
insertConcurrentStages(OpBuilder &builder, ArrayRef<Region *> stages,
                       MutableArrayRef<IRMapping> mappings) {
  SmallVector<unsigned> hoisted(stages.size(), 0);
  for (auto [index, stage] : llvm::enumerate(stages)) {
    if (index == 0)
      continue;
    for (Operation &op : stage->front().without_terminator()) {
      if (!isWaitForEarlierStage(op, *stage))
        break;
      builder.clone(op, mappings[index]);
      hoisted[index]++;
    }
  }

  SmallVector<SmallVector<Value>> results;
  for (auto [index, stage] : llvm::enumerate(stages))
    results.push_back(
        insertStage(builder, *stage, mappings[index], hoisted[index]));
  return results;
}

//===----------------------------------------------------------------------===//
// Prefetching the copy stage across iterations of the enclosing loop
//===----------------------------------------------------------------------===//
//
// A pipeline nested in a loop (e.g. the K pipeline of one output tile, in the
// loop over output tiles) starts every iteration with an on-ramp that only
// runs the copy stage: the compute cores wait for the first transfers of the
// iteration. Instead, the copy stage of the first pipeline iteration of
// enclosing iteration j + 1 is issued during the off-ramp of enclosing
// iteration j, concurrently with its last compute stage, and passed to
// iteration j + 1 through iter_args of the enclosing loop:
//
//   copy(lb) of j = lb'                 // prologue, before the loop
//   scf.for j iter_args(%r = ...) {
//     ...                               // e.g. copy the accumulator into L1
//     steady state (first step uses %r)
//     off-ramp: waits of the last compute stage
//               %n = scf.if (j + step < ub) { copy(lb) of j + step }
//               last compute stage
//     ...                               // e.g. copy the accumulator back
//     scf.yield %n
//   }
//
// The buffers of the copy stage are then used across iterations of the
// enclosing loop, so their allocations are hoisted out of it, into a
// `memref.alloca_scope` around it: allocations after the loop (e.g. those of
// a peeled remainder loop nest) can still reuse their memory.

/// Returns the value 'value' is a view of (through view-like ops and casts).
static Value getViewSource(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op))
      value = view.getViewSource();
    else if (auto cast = dyn_cast<memref::CastOp>(op))
      value = cast.getSource();
    else
      break;
  }
  return value;
}

/// Adds the view sources of the memrefs that 'op' (but not the ops nested in
/// it) may write to 'written'. Returns failure if they are unknown.
static LogicalResult collectWrittenMemRefs(Operation *op,
                                           DenseSet<Value> &written) {
  if (auto transfer = dyn_cast<dma::StartTransferOp>(op)) {
    written.insert(getViewSource(transfer.getDest()));
    return success();
  }
  if (auto zeroFill = dyn_cast<dma::StartZeroMemTransferOp>(op)) {
    written.insert(getViewSource(zeroFill.getFilled()));
    return success();
  }
  // Compute ops: may write any of their operands.
  if (isa<MemRefMicrokernelOp>(op)) {
    for (Value operand : op->getOperands())
      written.insert(getViewSource(operand));
    return success();
  }
  // Synchronization, or only the nested ops have effects.
  if (isa<BarrierOp, MicrokernelFenceOp, dma::WaitForTransferOp, PipelineOp>(
          op) ||
      op->hasTrait<OpTrait::IsTerminator>() ||
      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() || isPure(op))
    return success();
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return failure();
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  for (MemoryEffects::EffectInstance &instance : instances) {
    if (!isa<MemoryEffects::Write>(instance.getEffect()))
      continue;
    if (!instance.getValue())
      return failure();
    written.insert(getViewSource(instance.getValue()));
  }
  return success();
}

namespace {
/// The state of a pipeline whose copy stage is prefetched across iterations of
/// the enclosing loop.
struct CrossIterationPrefetch {
  /// The enclosing loop. Its last 'numValues' iter_args carry the results of
  /// the copy stage for the first pipeline iteration of the current iteration.
  scf::ForOp loop;
  unsigned numValues;
  /// The ops of the loop body (before the pipeline) that the copy stage uses,
  /// in order. They are pure and only depend on the induction variable of the
  /// loop and values defined outside of it.
  SmallVector<Operation *> dependencies;

  ValueRange getIterArgs() {
    return ValueRange(loop.getRegionIterArgs()).take_back(numValues);
  }
};
} // namespace

/// Inserts the copy stage of 'pipelineOp' for its first iteration, in the
/// iteration of the enclosing loop with induction variable 'loopIV'. Returns
/// the values the copy stage yields.
static SmallVector<Value> insertFirstCopyStage(OpBuilder &builder,
                                               PipelineOp pipelineOp,
                                               CrossIterationPrefetch &prefetch,
                                               Value loopIV) {
  IRMapping mapping;
  mapping.map(prefetch.loop.getInductionVar(), loopIV);
  for (Operation *op : prefetch.dependencies)
    builder.clone(*op, mapping);
  Region &stage = pipelineOp.getStages().front();
  mapping.map(stage.getArgument(0),
              mapping.lookupOrDefault(pipelineOp.getLowerBound()));
  return insertStage(builder, stage, mapping);
}

/// Checks whether the copy stage of 'pipelineOp' can be prefetched across
/// iterations of the enclosing loop and, if so, prepares it: hoists
/// 'resources' (the buffers of the pipeline, see 'makeResourceUsageExplicit')
/// out of the loop, inserts the copy stage for the first iteration of the loop
/// before it and adds the loop iter_args carrying the copy stage results.
/// 'onlyPipelineInLoop' must be true if the loop contains no other pipeline
/// (before any was lowered): otherwise the next copy would not overlap with
/// what precedes the pipeline in program order.
static std::optional<CrossIterationPrefetch>
prepareCrossIterationPrefetch(RewriterBase &rewriter, PipelineOp pipelineOp,
                              ArrayRef<memref::AllocaOp> resources,
                              bool onlyPipelineInLoop) {
  auto fail = [&](const Twine &reason) -> std::optional<CrossIterationPrefetch> {
    LLVM_DEBUG(llvm::dbgs() << "not prefetching across iterations: " << reason
                            << "\n");
    return std::nullopt;
  };

  auto loop = dyn_cast<scf::ForOp>(pipelineOp->getParentOp());
  if (!loop)
    return fail("pipeline is not directly nested in an scf.for");
  if (pipelineOp.getStages().size() != 2)
    return fail("only two-stage pipelines are supported");
  if (!onlyPipelineInLoop)
    return fail("enclosing loop contains other pipelines");

  // The loop must run at least twice, for there to be a next iteration.
  std::optional<llvm::APInt> loopTrips = loop.getStaticTripCount();
  if (!loopTrips || loopTrips->ult(2))
    return fail("enclosing loop does not have a static trip count >= 2");

  // The copy for the next iteration of the loop is issued while the last
  // compute stage runs. It writes the buffers of the first pipeline iteration,
  // which must not be those the last compute stage reads.
  std::optional<int64_t> lb = getConstantIntValue(pipelineOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(pipelineOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(pipelineOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return fail("pipeline bounds are not static");
  int64_t trips = llvm::divideCeil(*ub - *lb, *step);
  int64_t stages = pipelineOp.getStages().size();
  int64_t lastIV = *lb + (trips - 1) * *step;
  if (llvm::mod(llvm::divideFloorSigned(*lb, *step), stages) ==
      llvm::mod(llvm::divideFloorSigned(lastIV, *step), stages))
    return fail("first and last pipeline iteration use the same buffers");

  // The copy stage may only start transfers and compute pure values, so that
  // issuing it earlier, and conditionally, is fine.
  Region &copyStage = pipelineOp.getStages().front();
  SmallVector<Value> sources;
  WalkResult walk = copyStage.walk([&](Operation *op) {
    if (auto transfer = dyn_cast<dma::StartTransferOp>(op)) {
      sources.push_back(getViewSource(transfer.getSource()));
      return WalkResult::advance();
    }
    auto specialization = dyn_cast<CoreSpecializationOpInterface>(op);
    if (specialization && specialization.needsSynchronization())
      return WalkResult::interrupt();
    if (isa<dma::StartZeroMemTransferOp, dma::WaitForTransferOp,
            PipelineYieldOp>(op) ||
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() || isPure(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  if (walk.wasInterrupted())
    return fail("copy stage contains ops other than transfers and pure ops");

  // The transfers move ahead of everything else in the loop body (other than
  // the copy stage itself): nothing there may write the memory they read.
  DenseSet<Value> written;
  walk = loop.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (copyStage.isAncestor(op->getParentRegion()))
      return WalkResult::skip();
    if (failed(collectWrittenMemRefs(op, written)))
      return WalkResult::interrupt();
    // The effects of a microkernel's body are those of the microkernel.
    return isa<MemRefMicrokernelOp>(op) ? WalkResult::skip()
                                        : WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return fail("unknown memory writes in the enclosing loop");
  if (llvm::any_of(sources, [&](Value v) { return written.contains(v); }))
    return fail("the enclosing loop writes memory the copy stage reads");

  // Collect the ops of the loop body the copy stage depends on.
  DenseSet<Operation *> hoisted;
  for (memref::AllocaOp resource : resources)
    hoisted.insert(resource);
  SetVector<Value> used;
  getUsedValuesDefinedAbove(copyStage, copyStage, used);
  SmallVector<Value> worklist(used.begin(), used.end());
  worklist.push_back(pipelineOp.getLowerBound());
  SetVector<Operation *> dependencies;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (value == loop.getInductionVar())
      continue;
    if (!loop.getRegion().isAncestor(value.getParentRegion()))
      continue; // Defined above the loop.
    Operation *op = value.getDefiningOp();
    if (!op)
      return fail("copy stage depends on a loop-carried value");
    if (hoisted.contains(op) || dependencies.contains(op))
      continue;
    if (op->getBlock() != loop.getBody() || !isPure(op) ||
        op->getNumRegions() != 0)
      return fail("copy stage depends on a non-pure op in the loop");
    dependencies.insert(op);
    llvm::append_range(worklist, op->getOperands());
  }
  SmallVector<Operation *> orderedDependencies = dependencies.takeVector();
  llvm::sort(orderedDependencies, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  // Hoist the buffers out of the loop, into an alloca scope around it.
  Location loc = loop.getLoc();
  rewriter.setInsertionPoint(loop);
  auto scope =
      rewriter.create<memref::AllocaScopeOp>(loc, loop->getResultTypes());
  Block *scopeBody = rewriter.createBlock(&scope.getBodyRegion());
  rewriter.setInsertionPointToEnd(scopeBody);
  auto scopeReturn =
      rewriter.create<memref::AllocaScopeReturnOp>(loc, loop->getResults());
  for (auto [result, replacement] :
       llvm::zip_equal(loop->getResults(), scope.getResults()))
    rewriter.replaceAllUsesExcept(result, replacement, scopeReturn);
  loop->moveBefore(scopeReturn);
  for (memref::AllocaOp resource : resources)
    resource->moveBefore(loop);

  // Copy stage of the first iteration of the loop, before it.
  CrossIterationPrefetch prefetch{loop, 0, std::move(orderedDependencies)};
  rewriter.setInsertionPoint(loop);
  SmallVector<Value> initial = insertFirstCopyStage(
      rewriter, pipelineOp, prefetch, loop.getLowerBound());

  // Carry it into the loop. The yielded values are set by the off-ramp.
  FailureOr<LoopLikeOpInterface> newLoop = loop.replaceWithAdditionalYields(
      rewriter, initial, /*replaceInitOperandUsesInLoop=*/false,
      [](OpBuilder &, Location, ArrayRef<BlockArgument> iterArgs) {
        return SmallVector<Value>(iterArgs.begin(), iterArgs.end());
      });
  if (failed(newLoop)) {
    // Not expected for 'scf.for'; the IR is still valid, without prefetch.
    pipelineOp->emitError("failed to add iter_args to the enclosing loop");
    return std::nullopt;
  }
  prefetch.loop = cast<scf::ForOp>(*newLoop);
  prefetch.numValues = initial.size();
  return prefetch;
}

/// Inserts the copy stage of the first pipeline iteration of the next
/// iteration of the enclosing loop, if there is one, and yields its results to
/// that iteration.
static void insertNextCopyStage(OpBuilder &builder, PipelineOp pipelineOp,
                                CrossIterationPrefetch &prefetch) {
  scf::ForOp loop = prefetch.loop;
  Location loc = pipelineOp.getLoc();
  Value next = builder.create<arith::AddIOp>(loc, loop.getInductionVar(),
                                             loop.getStep());
  Value isNotLast = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
  auto ifOp = builder.create<scf::IfOp>(
      loc, TypeRange(prefetch.getIterArgs().getTypes()), isNotLast,
      /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    builder.create<scf::YieldOp>(
        loc, insertFirstCopyStage(builder, pipelineOp, prefetch, next));
    // The results of the last iteration are unused. Completed tokens (rather
    // than the incoming ones) let the iter_args fold away on the compute
    // cores, where all transfers are completed tokens.
    builder.setInsertionPointToStart(ifOp.elseBlock());
    SmallVector<Value> unused;
    for (Value iterArg : prefetch.getIterArgs())
      unused.push_back(isa<dma::TokenType>(iterArg.getType())
                           ? builder.create<dma::CompletedTokenOp>(loc)
                           : iterArg);
    builder.create<scf::YieldOp>(loc, unused);
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  unsigned first = yield.getNumOperands() - prefetch.numValues;
  for (auto [index, value] : llvm::enumerate(ifOp.getResults()))
    yield.setOperand(first + index, value);
}

/// Returns the local induction value for a given stage.
/// We separate between two kinds of induction values:
/// * The global induction value
/// * The local induction value of a stage
/// The global induction value is the induction value of the first pipeline
/// stage only.
/// Since the second stage with IV = 0 executes at the same time as the first
/// stage with IV = 1, a negative offset has to be applied to the global IV
/// to map it to a given stage. This returned value is the local induction
/// value.
/// This function generates the code necessary that maps the global IV
/// 'inductionVar' to the local IV of 'stage' and returns the local IV.
/// 'step' is the step value of the pipeline op.
static Value getInductionVarForStage(OpBuilder &builder, Region &stage,
                                     Value inductionVar, Value step) {
  return affine::makeComposedAffineApply(
      builder, inductionVar.getLoc(),
      builder.getAffineDimExpr(0) -
          builder.getAffineDimExpr(1) *
              builder.getAffineConstantExpr(stage.getRegionNumber()),
      {inductionVar, step});
}

void LowerPipelineOp::runOnOperation() {
  // Collected first: prefetching across iterations replaces the enclosing loop.
  SmallVector<PipelineOp> pipelineOps;
  getOperation()->walk([&](PipelineOp pipelineOp) {
    if (!pipelineOp.hasTensorSemantics())
      pipelineOps.push_back(pipelineOp);
  });
  // Whether a pipeline is the only one in its parent op, at any depth (decided
  // before lowering any of them).
  DenseMap<Operation *, bool> onlyPipelineInParent;
  for (PipelineOp pipelineOp : pipelineOps) {
    Operation *parent = pipelineOp->getParentOp();
    unsigned count = 0;
    parent->walk([&](PipelineOp) { count++; });
    onlyPipelineInParent[pipelineOp] = count == 1;
  }

  for (PipelineOp pipelineOp : pipelineOps) {
    SmallVector<memref::AllocaOp> resources =
        makeResourceUsageExplicit(pipelineOp);

    IRRewriter builder(pipelineOp);
    std::optional<CrossIterationPrefetch> prefetch;
    if (prefetchAcrossIterations)
      prefetch = prepareCrossIterationPrefetch(
          builder, pipelineOp, resources, onlyPipelineInParent[pipelineOp]);
    builder.setInsertionPoint(pipelineOp);

    Value currentIV = pipelineOp.getLowerBound();
    SmallVector<SmallVector<Value>> oldResultsOfStages(
        pipelineOp.getRegions().size() - 1);

    // Build on-ramp that executes all stages except the last.
    // As soon as we have started executing the last stage our pipeline is fully
    // utilized and we can switch to the 'scf.for'.
    for (unsigned i = 0; i < pipelineOp.getRegions().size() - 1; i++) {
      SmallVector<SmallVector<Value>> newResultsOfStages(
          pipelineOp.getRegions().size() - 1);

      // The first copy stage was issued by the previous iteration of the
      // enclosing loop (two stages: it is the whole on-ramp).
      if (prefetch) {
        newResultsOfStages[0] = llvm::to_vector(prefetch->getIterArgs());
        oldResultsOfStages = std::move(newResultsOfStages);
        currentIV = builder.create<arith::AddIOp>(
            pipelineOp->getLoc(), currentIV, pipelineOp.getStep());
        continue;
      }

      // Go over one more stage every iteration as the result of the previous
      // stage gets ready.
      SmallVector<Region *> stages;
      SmallVector<IRMapping> mappings;
      for (Region &stage : pipelineOp->getRegions().take_front(i + 1)) {
        IRMapping &mapping = mappings.emplace_back();
        stages.push_back(&stage);
        mapping.map(stage.getArguments().front(),
                    getInductionVarForStage(builder, stage, currentIV,
                                            pipelineOp.getStep()));
        if (&stage != &pipelineOp.getStages().front())
          for (auto [oldBlockArg, newBlockArg] :
               llvm::zip_equal(stage.getArguments().drop_front(),
                               oldResultsOfStages[stage.getRegionNumber() - 1]))
            mapping.map(oldBlockArg, newBlockArg);
      }
      for (auto [stage, results] : llvm::zip_equal(
               stages, insertConcurrentStages(builder, stages, mappings)))
        newResultsOfStages[stage->getRegionNumber()] = std::move(results);

      oldResultsOfStages = std::move(newResultsOfStages);
      currentIV = builder.create<arith::AddIOp>(pipelineOp->getLoc(), currentIV,
                                                pipelineOp.getStep());
    }

    SmallVector<Value> oldResultsOfStagesFlattened;
    // Maps from the index of a stage to the index of the first value in
    // 'oldResultsOfStagesFlattened' that is no longer part of this stages
    // result.
    SmallVector<size_t> endOfStageIndex{0};
    for (ArrayRef<Value> results : oldResultsOfStages) {
      llvm::append_range(oldResultsOfStagesFlattened, results);
      endOfStageIndex.push_back(endOfStageIndex.back() + results.size());
    }

    auto forOp = builder.create<scf::ForOp>(
        pipelineOp->getLoc(), currentIV, pipelineOp.getUpperBound(),
        pipelineOp.getStep(), oldResultsOfStagesFlattened);
    {
      OpBuilder::InsertionGuard guard{builder};
      builder.setInsertionPointToStart(forOp.getBody());

      SmallVector<Region *> stages;
      SmallVector<IRMapping> mappings;
      for (Region &stage : pipelineOp.getStages()) {
        Value iv = getInductionVarForStage(
            builder, stage, forOp.getInductionVar(), pipelineOp.getStep());

        IRMapping &mapping = mappings.emplace_back();
        stages.push_back(&stage);
        mapping.map(stage.getArguments().front(), iv);
        if (&stage != &pipelineOp.getStages().front()) {
          size_t begin = endOfStageIndex[stage.getRegionNumber() - 1];
          size_t end = endOfStageIndex[stage.getRegionNumber()];
          MutableArrayRef<BlockArgument> argument =
              forOp.getRegionIterArgs().slice(begin, end - begin);
          for (auto [oldBlockArg, newBlockArg] :
               llvm::zip_equal(stage.getArguments().drop_front(), argument))
            mapping.map(oldBlockArg, newBlockArg);
        }
      }
      SmallVector<Value> newResultsOfStagesFlattened;
      for (SmallVector<Value> &results :
           insertConcurrentStages(builder, stages, mappings))
        llvm::append_range(newResultsOfStagesFlattened, results);
      builder.create<scf::YieldOp>(pipelineOp->getLoc(),
                                   newResultsOfStagesFlattened);
    }

    // Off-ramp.
    for (unsigned i = 0; i < endOfStageIndex.size() - 1; i++) {
      uint64_t begin = endOfStageIndex[i];
      uint64_t end = endOfStageIndex[i + 1];
      oldResultsOfStages[i] = forOp.getResults().slice(begin, end - begin);
    }

    // Compute the IV as it'd be at the end of the 'scf.for', i.e. one step past
    // the last iteration: lb + ceildiv(ub - lb, step) * step. Rounding up
    // matters when the trip range is not a multiple of the step (e.g. padded
    // remainder tiles), otherwise the off-ramp would reuse an earlier IV.
    AffineExpr lb = builder.getAffineDimExpr(0);
    AffineExpr ub = builder.getAffineDimExpr(1);
    AffineExpr step = builder.getAffineDimExpr(2);
    currentIV = affine::makeComposedAffineApply(
        builder, pipelineOp->getLoc(), lb + (ub - lb).ceilDiv(step) * step,
        {pipelineOp.getLowerBound(), pipelineOp.getUpperBound(),
         pipelineOp.getStep()});

    // Paste the stages until the last stage has done its last iteration.
    for (unsigned i = 1; i < pipelineOp.getRegions().size(); i++) {
      SmallVector<SmallVector<Value>> newResultsOfStages(
          pipelineOp.getRegions().size() - 1);

      // One fewer stage every iteration, first stage already completed.
      SmallVector<Region *> stages;
      SmallVector<IRMapping> mappings;
      for (Region &stage : pipelineOp->getRegions().drop_front(i)) {
        Value iv = getInductionVarForStage(builder, stage, currentIV,
                                           pipelineOp.getStep());

        IRMapping &mapping = mappings.emplace_back();
        stages.push_back(&stage);
        mapping.map(stage.getArguments().front(), iv);
        for (auto [oldBlockArg, newBlockArg] :
             llvm::zip_equal(stage.getArguments().drop_front(),
                             oldResultsOfStages[stage.getRegionNumber() - 1]))
          mapping.map(oldBlockArg, newBlockArg);
      }
      // Last step, the last compute stage alone (two stages): issue the copy
      // stage of the next iteration of the enclosing loop concurrently, after
      // the waits for its own transfers.
      if (prefetch) {
        Region &stage = *stages.front();
        unsigned waits = 0;
        for (Operation &op : stage.front().without_terminator()) {
          if (!isWaitForEarlierStage(op, stage))
            break;
          builder.clone(op, mappings.front());
          waits++;
        }
        insertNextCopyStage(builder, pipelineOp, *prefetch);
        insertStage(builder, stage, mappings.front(), waits);
        continue;
      }

      for (auto [stage, results] : llvm::zip_equal(
               stages, insertConcurrentStages(builder, stages, mappings)))
        if (stage != &pipelineOp.getStages().back())
          newResultsOfStages[stage->getRegionNumber()] = std::move(results);
      oldResultsOfStages = std::move(newResultsOfStages);
      currentIV = builder.create<arith::AddIOp>(pipelineOp->getLoc(), currentIV,
                                                pipelineOp.getStep());
    }

    pipelineOp->erase();
  }
}
