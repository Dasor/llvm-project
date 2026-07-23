#include "mlir/Dialect/Snitch/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::snitch {
#define GEN_PASS_DEF_SPECIALIZEDMACODEPASS
#include "mlir/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace mlir::snitch

namespace {
class SpecializeDMACode
    : public mlir::snitch::impl::SpecializeDMACodePassBase<
          SpecializeDMACode> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;

private:
};

} // namespace

using namespace mlir;
using namespace mlir::snitch;

/// Removes all operations from 'function' that implement
/// 'CoreSpecializationOpInterface' but not 'Interface'.
template <typename Interface>
static void removeUnsupportedSpecializedOps(FunctionOpInterface function) {
  function->walk([&](CoreSpecializationOpInterface operation) {
    if (isa<Interface>(*operation))
      return;

    IRRewriter rewriter(operation);
    operation.replaceWithNoop(rewriter);
  });
}

/// Inserts a barrier after every operation requiring according to
/// 'CoreSpecializationOpInterface'.
/// Note: Does not currently support barriers in divergent control flow.
static void insertBarriers(FunctionOpInterface function) {
  function->walk([](CoreSpecializationOpInterface operation) {
    if (!operation.needsSynchronization())
      return;

    OpBuilder builder(operation.getContext());
    builder.setInsertionPointAfter(operation);
    builder.create<BarrierOp>(operation->getLoc());
  });
}

void SpecializeDMACode::runOnOperation() {
  auto *dialect = getContext().getLoadedDialect<SnitchDialect>();
  SymbolTable table(getOperation());
  auto toSpecialize =
      llvm::to_vector(getOperation().getOps<FunctionOpInterface>());
  for (FunctionOpInterface function : toSpecialize) {
    if (function.isDeclaration())
      continue;

    insertBarriers(function);

    FunctionOpInterface clone = function.clone();
    clone.setName((clone.getName() + "$dma").str());
    table.insert(clone, std::next(function->getIterator()));
    dialect->getDmaSpecializationAttrHelper().setAttr(
        function, FlatSymbolRefAttr::get(clone));

    removeUnsupportedSpecializedOps<ComputeCoreSpecializationOpInterface>(
        function);
    removeUnsupportedSpecializedOps<DMACoreSpecializationOpInterface>(clone);
  }
}
