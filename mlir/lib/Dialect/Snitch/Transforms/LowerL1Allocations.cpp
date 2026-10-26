#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Snitch/IR/SnitchAttrs.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_LOWERL1ALLOCATIONSPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class LowerL1Allocations
    : public mlir::snitch::impl::LowerL1AllocationsPassBase<
          LowerL1Allocations> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace mlir::snitch;

/// Returns the byte alignment of 'allocOp'.
static uint64_t getAlignment(memref::AllocaOp allocOp) {
  // Note: This assumes bitWidth == alignment == size.
  if (std::optional<uint64_t> alignment = allocOp.getAlignment())
    return *alignment;
  return llvm::divideCeil(allocOp.getType().getElementTypeBitWidth(), 8);
}

/// Returns the number of elements needed to back 'memRefType' (which may
/// include padding elements due to its strides), or failure if its layout is
/// unsupported.
static FailureOr<int64_t> getAllocElements(MemRefType memRefType,
                                           SmallVectorImpl<int64_t> &strides) {
  int64_t offset;
  if (failed(memRefType.getStridesAndOffset(strides, offset)) || offset != 0)
    return failure();
  // Compute this via the linearized access of the last element + 1.
  int64_t allocElements = 1;
  for (auto [stride, shape] : llvm::zip_equal(strides, memRefType.getShape()))
    allocElements += stride * (shape - 1);
  return allocElements;
}

static uint64_t getAllocBytes(memref::AllocaOp allocOp) {
  SmallVector<int64_t> strides;
  FailureOr<int64_t> elements = getAllocElements(allocOp.getType(), strides);
  return llvm::divideCeil(allocOp.getType().getElementTypeBitWidth(), 8) *
         (succeeded(elements) ? *elements : 0);
}

/// Returns true if the memory of 'allocOp' may be used after its parent block
/// has finished executing: i.e. if a value derived from it (through views and
/// the results of ops yielding it) is yielded by the terminator of that block.
static bool mayEscapeBlock(memref::AllocaOp allocOp) {
  Block *block = allocOp->getBlock();
  SmallVector<Value> worklist{allocOp.getResult()};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (user->hasTrait<OpTrait::IsTerminator>()) {
        if (user->getBlock() == block)
          return true;
        // Conservatively, the yielded value may be any result of the op the
        // terminator returns to.
        if (Operation *parent = user->getParentOp())
          llvm::append_range(worklist, parent->getResults());
        continue;
      }
      if (isa<ViewLikeOpInterface, memref::CastOp>(user))
        llvm::append_range(worklist, user->getResults());
    }
  }
  return false;
}

/// Assigns L1 offsets to the allocations in 'block' and the regions nested in
/// it, starting at 'base'. Returns the end of the memory used.
///
/// An allocation directly in 'block' is live from its definition to the end of
/// the block (allocations do not escape their block, checked by the caller).
/// The ops of a block execute one after another, so the allocations nested in
/// an op only overlap in time with the block's allocations defined before that
/// op: they are placed above those, and an allocation defined after the op can
/// reuse the op's memory. The regions of one op share memory as well (they are
/// alternatives or run one after another).
static uint64_t
assignOffsets(Block &block, uint64_t base,
              const DenseSet<Operation *> &l1Allocs,
              DenseMap<Operation *, uint64_t> &offsets) {
  uint64_t live = base; // End of the block's allocations defined so far.
  uint64_t end = base;
  for (Operation &op : block) {
    auto allocOp = dyn_cast<memref::AllocaOp>(op);
    if (allocOp && l1Allocs.contains(allocOp)) {
      live = llvm::alignTo(live, getAlignment(allocOp));
      offsets[allocOp] = live;
      live += getAllocBytes(allocOp);
      end = std::max(end, live);
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        end = std::max(end, assignOffsets(nested, live, l1Allocs, offsets));
  }
  return end;
}

void LowerL1Allocations::runOnOperation() {
  SmallVector<memref::AllocaOp> allocs;
  getOperation()->walk([&](memref::AllocaOp allocOp) {
    if (!isa_and_nonnull<L1EncodingAttr>(allocOp.getType().getMemorySpace()))
      return;
    if (!allocOp.getDynamicSizes().empty()) {
      // Note: There is no reason for this being unsupported other than we don't
      // need it right now and its extra work.
      allocOp->emitOpError(
          "L1 allocations with dynamic size is currently unsupported");
      signalPassFailure();
      return;
    }

    allocs.push_back(allocOp);
  });
  if (allocs.empty())
    return;

  // Plan the offsets. Allocations in different, sequentially executed parts
  // of the function (e.g. the loop nests of peeled remainder tiles) share
  // memory, unless an allocation may outlive its block, in which case every
  // allocation gets its own memory.
  DenseMap<Operation *, uint64_t> offsets;
  uint64_t end = 0;
  auto escaping = llvm::find_if(allocs, mayEscapeBlock);
  if (escaping != allocs.end())
    escaping->emitRemark("L1 allocation may outlive its block; not reusing "
                         "L1 memory between sequential loop nests");
  if (escaping == allocs.end()) {
    DenseSet<Operation *> l1Allocs(allocs.begin(), allocs.end());
    for (Block &block : getOperation().getFunctionBody())
      end = std::max(end, assignOffsets(block, 0, l1Allocs, offsets));
  } else {
    for (memref::AllocaOp allocOp : allocs) {
      end = llvm::alignTo(end, getAlignment(allocOp));
      offsets[allocOp] = end;
      end += getAllocBytes(allocOp);
    }
  }

  if (end >= l1MemoryBytes) {
    auto diagEmit =
        assertCompiled ? &Operation::emitError : &Operation::emitWarning;
    ((*getOperation()).*
     diagEmit)("kernel does not fit into L1 memory and cannot be compiled");
    if (assertCompiled) {
      signalPassFailure();
      return;
    }

    auto *dialect = getContext().getLoadedDialect<SnitchDialect>();
    dialect->getXdslCompilationFailedAttrHelper().setAttr(
        getOperation(), UnitAttr::get(&getContext()));

    // The function is in an invalid state now that we cannot lower. Work
    // around this by erasing the body completely.
    getOperation().setPrivate();
    getOperation().getBlocks().clear();
    return;
  }

  auto builder = OpBuilder::atBlockBegin(&getOperation().front());
  auto l1Memory = builder.create<L1MemoryViewOp>(
      getOperation().getLoc(),
      MemRefType::get({l1MemoryBytes}, builder.getI8Type()));
  for (memref::AllocaOp allocOp : allocs) {
    builder.setInsertionPoint(allocOp);
    MemRefType memRefType = allocOp.getType();
    auto byteShift = builder.create<arith::ConstantIndexOp>(
        allocOp.getLoc(), offsets.lookup(allocOp));

    // We do not support anything but a zero offset right now.
    SmallVector<int64_t> strides;
    FailureOr<int64_t> allocElements = getAllocElements(memRefType, strides);
    if (failed(allocElements)) {
      allocOp->emitOpError("Cannot lower MemRef in L1 memory with a "
                           "non-strided layout or a non-zero offset");
      signalPassFailure();
      return;
    }

    // First, allocate one large contiguous element memref.
    // Get rid of the memory space at this point as well.
    Value view = builder.create<memref::ViewOp>(
        allocOp.getLoc(),
        MemRefType::get({*allocElements}, memRefType.getElementType()), l1Memory,
        byteShift,
        /*sizes=*/ValueRange());

    // Reinterpret cast the view with the actual shape and strides.
    view = builder.create<memref::ReinterpretCastOp>(
        allocOp.getLoc(),
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        memRefType.getLayout()),
        view, 0, memRefType.getShape(), strides);
    allocOp.replaceAllUsesWith(view);
    allocOp.erase();
  }

  // Alloca scopes (e.g. from 'snitch-lower-pipeline-op') only limited the
  // lifetime of L1 allocations; without allocations left they are no-ops.
  // Inline them: canonicalization does not when they contain ops with unknown
  // memory effects, and later lowerings do not support them around
  // multi-block control flow.
  SmallVector<memref::AllocaScopeOp> scopes;
  getOperation()->walk([&](memref::AllocaScopeOp scope) {
    if (!scope->walk([](memref::AllocaOp) { return WalkResult::interrupt(); })
             .wasInterrupted())
      scopes.push_back(scope);
  });
  for (memref::AllocaScopeOp scope : scopes) {
    IRRewriter rewriter(scope);
    Block *body = &scope.getBodyRegion().front();
    Operation *terminator = body->getTerminator();
    SmallVector<Value> results(terminator->getOperands());
    rewriter.eraseOp(terminator);
    rewriter.inlineBlockBefore(body, scope);
    rewriter.replaceOp(scope, results);
  }

  // Change any leftover memory space occurrences.
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](MemRefType memRefType) -> std::optional<MemRefType> {
        if (!memRefType.getMemorySpace())
          return std::nullopt;

        return MemRefType::get(memRefType.getShape(),
                               memRefType.getElementType(),
                               memRefType.getLayout());
      });
  replacer.recursivelyReplaceElementsIn(getOperation(), /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}
