#ifndef MLIR_DIALECT_SNITCH_IR_SNITCHINTERFACES
#define MLIR_DIALECT_SNITCH_IR_SNITCHINTERFACES

include "mlir/IR/Interfaces.td"

def Snitch_CoreSpecializationOpInterface
  : OpInterface<"CoreSpecializationOpInterface"> {
  let cppNamespace = "::mlir::snitch";

  let description = [{
    Interface used as a base class for ops meant to only run on a specific core.
    When specializing a function for a specific core, ops implementing this
    interface but not supported on a specific core will be removed using
    `replaceWithNoop`.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method called to replace this operation with a noop in an unsupported
        specialization. `rewriter`s insertion point is set right before the
        operation.

        The op must have been erased when this method returns.
      }],
      /*retTy=*/"void",
      /*methodName=*/"replaceWithNoop",
      /*args=*/(ins "mlir::RewriterBase&":$rewriter)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation requires synchronization between all cores.
      }],
      "bool", "needsSynchronization", (ins), [{}], [{
        return false;
      }]
    >
  ];
}

def Snitch_DMACoreSpecializationOpInterface
  : OpInterface<"DMACoreSpecializationOpInterface", [Snitch_CoreSpecializationOpInterface]> {
  let cppNamespace = "::mlir::snitch";
}

def Snitch_ComputeCoreSpecializationOpInterface
  : OpInterface<"ComputeCoreSpecializationOpInterface", [Snitch_CoreSpecializationOpInterface]> {
  let cppNamespace = "::mlir::snitch";
}

#endif // MLIR_DIALECT_SNITCH_IR_SNITCHINTERFACES
