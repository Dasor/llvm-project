#include "mlir/Conversion/SnitchToLLVM/SnitchToLLVM.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Snitch/IR/SnitchDialect.h"
#include "mlir/Dialect/Snitch/IR/SnitchOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSNITCHTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::snitch;

namespace {
struct L1MemoryViewOpLowering : ConvertOpToLLVMPattern<L1MemoryViewOp> {
  using ConvertOpToLLVMPattern<L1MemoryViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(L1MemoryViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    this->getMemRefDescriptorSizes(op->getLoc(), op.getType(),
                                   adaptor.getOperands(), rewriter, sizes,
                                   strides, size);

    // TODO: This is horribly hardcoded when it shouldn't be.
    Value l1Address = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(0x10000000));
    Value allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(
        op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(), l1Address);

    auto memRefDescriptor =
        this->createMemRefDescriptor(op->getLoc(), op.getType(), allocatedPtr,
                                     allocatedPtr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};

struct BarrierOpLowering : ConvertOpToLLVMPattern<BarrierOp> {

  LLVM::GlobalOp barrierGlobal;
  LLVM::LLVMFuncOp partialBarrierFunc;
  unsigned numParticipants;

  BarrierOpLowering(LLVM::GlobalOp barrierGlobal,
                    LLVM::LLVMFuncOp partialBarrierFunc,
                    unsigned numParticipants,
                    const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), barrierGlobal(barrierGlobal),
        partialBarrierFunc(partialBarrierFunc),
        numParticipants(numParticipants) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Every snitch.barrier is a rendezvous between exactly the compute-core
    // and DM-core clones SpecializeDMACode produces - not a whole-cluster
    // barrier. Lower it to a 2-party software barrier (snrt_partial_barrier)
    // against the runtime's existing, otherwise-idle _snrt_barrier global,
    // instead of the all-9-hart hardware CSR barrier: that way harts with no
    // work of their own never need to participate, and no barrier count ever
    // needs to be known outside this pass.
    Location loc = op.getLoc();
    Value barrierPtr = rewriter.create<LLVM::AddressOfOp>(loc, barrierGlobal);
    Value n = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(numParticipants));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, partialBarrierFunc,
                                              ValueRange{barrierPtr, n});
    return success();
  }
};

struct MicrokernelFenceOpLowering : ConvertOpToLLVMPattern<MicrokernelFenceOp> {

  using ConvertOpToLLVMPattern<MicrokernelFenceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MicrokernelFenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Sync up the FPU and integer pipelines by creating a (fake) data
    // dependency between an FPU and integer register.
    rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), /*res=*/rewriter.getI32Type(),
        /*operands=*/ValueRange(),
        "fmv.x.w $0, fa0\n"
        "mv $0, $0",
        // Tell the register allocator to allocate `$0` as if returning a
        // register. Also consider it to clob memory given that all
        // side effects of the FPU pipeline only become visible after this
        // instruction.
        /*constraints=*/"=r,~{memory}",
        // 'has_side_effects' is currently set to true due to a bug in MLIR
        // DCEing despite the memory clobber.
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::tailcallkind::TailCallKind::None,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CallMicrokernelOpLowering : ConvertOpToLLVMPattern<CallMicrokernelOp> {
  mutable SymbolTable symbolTable;

  CallMicrokernelOpLowering(SymbolTable symbolTable,
                            const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), symbolTable(std::move(symbolTable)) {
  }

  LogicalResult
  matchAndRewrite(CallMicrokernelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMFuncOp kernelDecl;
    {
      OpBuilder::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToEnd(
          &symbolTable.getOp()->getRegion(0).back());

      SmallVector<Type> types;
      for (Type type : op.getInputs().getTypes()) {
        if (auto memRefType = dyn_cast<MemRefType>(type)) {
          // Pretend layouts don't exit.
          type = MemRefType::get(
              memRefType.getShape(), memRefType.getElementType(),
              /*layout=*/nullptr, memRefType.getMemorySpace());
        }
        SmallVector<Type> converted;
        if (failed(getTypeConverter()->convertCallingConventionType(
                type, converted, /*useBarePointerCallConv=*/true)) ||
            converted.size() != 1)
          return failure();

        types.push_back(converted.front());
      }

      kernelDecl = rewriter.create<LLVM::LLVMFuncOp>(
          op.getLoc(), op.getName(),
          LLVM::LLVMFunctionType::get(rewriter.getType<LLVM::LLVMVoidType>(),
                                      types));
      symbolTable.insert(kernelDecl);

      // Required to tell the conversion pass to LLVM that this is actually a
      // call into the same linkage unit, and does not have to be rewritten to a
      // HAL module call.
      kernelDecl->setAttr("hal.import.bitcode", rewriter.getUnitAttr());

      cast<SnitchDialect>(op->getDialect())
          ->getRiscvAssemblyAttrHelper()
          .setAttr(kernelDecl, op.getRiscvAssemblyAttr());
    }

    SmallVector<Value> inputs;
    for (auto [value, oldType] :
         llvm::zip_equal(adaptor.getInputs(), op.getInputs().getType())) {
      auto memRefType = dyn_cast<MemRefType>(oldType);
      if (!memRefType) {
        inputs.push_back(value);
        continue;
      }
      auto descriptor = MemRefDescriptor(value);
      inputs.push_back(descriptor.bufferPtr(rewriter, op->getLoc(),
                                            *getTypeConverter(), memRefType));
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, kernelDecl, inputs);
    return success();
  }
};

struct ComputeCoreIndexOpLowering : ConvertOpToLLVMPattern<ComputeCoreIndexOp> {

  LLVM::LLVMFuncOp computeCoreIndexFunc;

  ComputeCoreIndexOpLowering(LLVM::LLVMFuncOp computeCoreIndexFunc,
                             const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter),
        computeCoreIndexFunc(computeCoreIndexFunc) {}

  LogicalResult
  matchAndRewrite(ComputeCoreIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, computeCoreIndexFunc,
                                              ValueRange());
    return success();
  }
};

} // namespace

void mlir::populateSnitchToLLVMConversionPatterns(
    mlir::ModuleOp moduleOp, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, unsigned barrierParticipants) {

  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
  IntegerType i32 = builder.getI32Type();
  auto computeCoreIndex = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_core_idx",
      LLVM::LLVMFunctionType::get(i32, ArrayRef<Type>{}));
  computeCoreIndex->setAttr("hal.import.bitcode", builder.getUnitAttr());

  // `_snrt_barrier` (snitch_cluster/sw/snRuntime/api/sync_decls.h):
  //   extern volatile struct { uint32_t cnt; uint32_t iteration; } _snrt_barrier;
  // Declared here as an external (no initializer) global purely so
  // BarrierOpLowering can take its address - the real definition lives in
  // sync.c and is resolved at link time against libsnRuntime.a.
  auto barrierStructTy =
      LLVM::LLVMStructType::getLiteral(builder.getContext(), {i32, i32});
  auto barrierGlobal = builder.create<LLVM::GlobalOp>(
      builder.getUnknownLoc(), barrierStructTy, /*isConstant=*/false,
      LLVM::Linkage::External, "_snrt_barrier", /*value=*/Attribute());

  auto partialBarrierFunc = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_partial_barrier",
      LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(builder.getContext()),
          ArrayRef<Type>{LLVM::LLVMPointerType::get(builder.getContext()),
                        i32}));
  partialBarrierFunc->setAttr("hal.import.bitcode", builder.getUnitAttr());

  patterns.insert<L1MemoryViewOpLowering, MicrokernelFenceOpLowering>(
      typeConverter);
  patterns.insert<BarrierOpLowering>(barrierGlobal, partialBarrierFunc,
                                     barrierParticipants, typeConverter);
  patterns.insert<ComputeCoreIndexOpLowering>(computeCoreIndex, typeConverter);
  patterns.insert<CallMicrokernelOpLowering>(SymbolTable(moduleOp),
                                             typeConverter);
}

namespace {
struct ConvertSnitchToLLVMPass
    : public mlir::impl::ConvertSnitchToLLVMPassBase<ConvertSnitchToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertSnitchToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Snitch is a fixed RV32 target: its pointers/sizes are always 32 bits.
  LowerToLLVMOptions options(&getContext());
  options.overrideIndexBitwidth(32);
  LLVMTypeConverter typeConverter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  populateSnitchToLLVMConversionPatterns(module, typeConverter, patterns,
                                         barrierParticipants);

  LLVMConversionTarget target(getContext());
  target.addIllegalDialect<SnitchDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
