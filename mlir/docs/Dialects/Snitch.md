# 'snitch' Dialect

[TOC]

The `snitch` dialect provides the core primitives used to lower tiled
`linalg` compute kernels onto the ETH Zürich/PULP **Snitch** RISC-V compute
cluster: marking regions of code as *microkernels* to be compiled separately
(by [xDSL](https://github.com/xdslproject/xdsl)) to hand-optimized
Snitch-stream RISC-V assembly, giving those microkernels a view over the
cluster's L1 scratchpad memory, specializing a single function body into a
DMA-core clone and a compute-core clone with the necessary barriers between
them, distributing work across the cluster's compute cores, and software
pipelining (dual-buffering) DMA transfers against compute.

A function is lowered through this dialect roughly in the following order:

1. `linalg` ops with tensor semantics are wrapped in `snitch.tensor.microkernel`
   regions (`-snitch-form-microkernels`).
2. Operands/results are promoted into L1 scratchpad memory via the `dma`
   dialect (`-snitch-promote-pads-to-l1`, `-snitch-promote-operands-to-l1`,
   `-snitch-promote-allocs-to-l1`).
3. Optionally, a tiled loop marked for double buffering is rewritten into a
   two-stage `snitch.pipeline` (`-snitch-pipeline-copy-compute`).
4. The IR is bufferized (`-snitch-eliminate-empty-tensors`,
   `-snitch-bufferize`), turning `snitch.tensor.microkernel` into
   `snitch.memref.microkernel` and DMA-promoted tensors into `memref.alloca`s.
5. `scf.forall` loops from thread-level tiling are lowered to per-hart
   strided `scf.for` loops (`-snitch-lower-forall-op`), and `snitch.pipeline`
   is expanded into an explicit on-ramp/steady-state/off-ramp loop nest
   (`-snitch-lower-pipeline-op`).
6. L1 allocations are assigned static byte offsets into a single
   `snitch.l1_memory_view` (`-snitch-lower-l1-allocations`).
7. The function is cloned into a DMA-core body and a compute-core body, with
   `snitch.barrier`s inserted at synchronization points
   (`-snitch-specialize-dma-code`).
8. `snitch.memref.microkernel` bodies are compiled to RISC-V assembly via
   xDSL and replaced with `snitch.call_microkernel` (`--convert-to-riscv`,
   a [Conversion pass](#conversion-passes)).
9. Everything else is lowered to LLVM (`--convert-snitch-to-llvm`, also a
   [Conversion pass](#conversion-passes)).

See the [`dma` dialect](DMA.md) for the tensor/memref copy operations used to
stage data into L1, and the [`snitch_dma` dialect](SnitchDMA.md) for the
lower-level primitives `dma` transfers are eventually legalized into.

## Attributes

### L1EncodingAttr

Syntax: `#snitch.l1_encoding`

Attribute used as memory space on a `memref` to denote it being in L1 memory.

## Interfaces

### CoreSpecializationOpInterface (`Snitch_CoreSpecializationOpInterface`)

Interface used as a base class for ops meant to only run on a specific core.
When specializing a function for a specific core, ops implementing this
interface but not supported on a specific core will be removed using
`replaceWithNoop`.

Methods:

- `void replaceWithNoop(mlir::RewriterBase &rewriter)` — replaces this
  operation with a noop in an unsupported specialization. The rewriter's
  insertion point is set right before the operation. The op must have been
  erased when this method returns.
- `bool needsSynchronization()` — returns true if this operation requires
  synchronization between all cores. When true, `-snitch-specialize-dma-code`
  inserts a `snitch.barrier` immediately after the op.

### ComputeCoreSpecializationOpInterface (`Snitch_ComputeCoreSpecializationOpInterface`)

Extends `CoreSpecializationOpInterface`. Marks ops that should survive on the
compute-core clone of a function produced by `-snitch-specialize-dma-code`.
Implemented by `snitch.memref.microkernel`, `snitch.microkernel_fence`, and
`snitch.compute_core_index`.

### DMACoreSpecializationOpInterface (`Snitch_DMACoreSpecializationOpInterface`)

Extends `CoreSpecializationOpInterface`. Marks ops that should survive on the
DMA-core clone of a function produced by `-snitch-specialize-dma-code`.
Implemented by the DMA-dialect transfer ops — see [DMA.md](DMA.md#extensions).

## Operations

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Snitch/IR/SnitchOps.td)

### `snitch.barrier` (snitch::BarrierOp)

Syntax:

```
operation ::= `snitch.barrier` attr-dict
```

Synchronization barrier across all cores participating in a kernel (DMA core
and compute core). Inserted automatically by `-snitch-specialize-dma-code`
after any op whose `needsSynchronization()` returns true. Lowered by
`--convert-snitch-to-llvm` to a call to `snrt_partial_barrier`, a
software barrier sized by that pass's `barrier-participants` option (default
2 — one DMA core and one compute core), not the full hardware barrier.

Example:

```mlir
snitch.barrier
```

### `snitch.call_microkernel` (snitch::CallMicrokernelOp)

Syntax:

```
operation ::= `snitch.call_microkernel` $name `(` $inputs `)` (`:` type($inputs)^)? `[``{`
              custom<RISCVAssembly>($riscv_assembly)
              `}` `]` attr-dict
```

Operation denoting a call to a compiled microkernel.
The compiled artifact is available as the `riscv_assembly` attribute.
`name` is used as a hint for the symbol name of the microkernel and has no
semantics.

The Microkernel may be executed asynchronous.
Side-effects of operations are therefore only guaranteed to be visible
after a subsequent invocation of `microkernel_fence`.

This is the op `--convert-to-riscv` produces in place of a
`snitch.memref.microkernel`, embedding the RISC-V assembly text produced by
`xdsl-opt`.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>riscv_assembly</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of any non-token type |

Example:

```mlir
snitch.call_microkernel "matmul"(%a, %b, %c) : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32> [{
  "add a0, a1, a2"
  "csrr a3, mcycle"
}]
```

### `snitch.compute_core_index` (snitch::ComputeCoreIndexOp)

Syntax:

```
operation ::= `snitch.compute_core_index` attr-dict
```

Returns the index of the compute core within a given cluster.
This is guaranteed to return a number between 0 and exclusive
`compute_cores` where `compute_cores` is an `IntegerAttr` in
the surrounding target attribute.

Used by `-snitch-lower-forall-op` to build the strided per-hart loop that
distributes `scf.forall` iterations across compute cores. When
`-snitch-specialize-dma-code` clones a function for the DMA core, its
`replaceWithNoop` implementation replaces this op with
`arith.constant 0 : index`, so the DMA clone follows compute-core-0's control
flow.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `Snitch_ComputeCoreSpecializationOpInterface`, `Snitch_CoreSpecializationOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | index |

Example:

```mlir
%id = snitch.compute_core_index
```

### `snitch.l1_memory_view` (snitch::L1MemoryViewOp)

Syntax:

```
operation ::= `snitch.l1_memory_view` `->` type($result) attr-dict
```

Produces a flat byte-addressed view over the cluster's L1 scratchpad.
Individual allocations are carved out of it by `-snitch-lower-l1-allocations`,
which assigns each L1-encoded `memref.alloca` a static offset into this
buffer. Lowered by `--convert-snitch-to-llvm` to a `memref` descriptor
pointing at the hardcoded L1 SPM base address.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | one-dimensional i8 MemRef of a static size |

Example:

```mlir
%l1 = snitch.l1_memory_view -> memref<112640xi8>
```

### `snitch.memref.microkernel` (snitch::MemRefMicrokernelOp)

Syntax:

```
operation ::= `snitch.memref.microkernel` `` `(` $inputs `)` (`:` type($inputs)^)? $body attr-dict
```

Operation denoting a region of operations as a microkernel.
The region is `IsolatedFromAbove` making all inputs to the microkernel explicit.
A later compilation step turns the "uncompiled" microkernel into a compiled
`call_microkernel` operation.

Operations within the Microkernel may be executed asynchronous.
Side-effects of operations are therefore only guaranteed to be visible
after a subsequent invocation of `microkernel_fence`.

The post-bufferization counterpart of `snitch.tensor.microkernel`; produced
by `-snitch-bufferize`, consumed by `--convert-to-riscv`.

Traits: `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`

Interfaces: `Snitch_ComputeCoreSpecializationOpInterface`, `Snitch_CoreSpecializationOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of any non-token type |

Example:

```mlir
snitch.memref.microkernel(%a, %b) : memref<4x4xf32>, memref<4x4xf32> {
^bb0(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>):
  linalg.generic ...
}
```

### `snitch.microkernel_fence` (snitch::MicrokernelFenceOp)

Syntax:

```
operation ::= `snitch.microkernel_fence` attr-dict
```

Execution of this operation guarantees that the side-effects of all
previous microkernel invocations are visible as soon as this operation
returns. `needsSynchronization()` is overridden to return `true`, so
`-snitch-specialize-dma-code` inserts a `snitch.barrier` immediately after
it. Lowered by `--convert-snitch-to-llvm` to an inline-asm sequence that
creates a fake register dependency to force pipeline synchronization.

Interfaces: `Snitch_ComputeCoreSpecializationOpInterface`, `Snitch_CoreSpecializationOpInterface`

Example:

```mlir
snitch.microkernel_fence
```

### `snitch.microkernel_yield` (snitch::MicrokernelYieldOp)

Syntax:

```
operation ::= `snitch.microkernel_yield` $results (`:` type($results)^)? attr-dict
```

Terminator of `snitch.tensor.microkernel`, yielding its tensor results.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<TensorMicrokernelOp>`, `ReturnLike`, `Terminator`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `results` | variadic of ranked tensor of any non-token type values |

Example:

```mlir
snitch.microkernel_yield %sum, %max : tensor<4x4xf32>, tensor<4xf32>
```

### `snitch.pipeline` (snitch::PipelineOp)

Syntax:

```
operation ::= `snitch.pipeline` $lower_bound `to` $upper_bound `step` $step (`inits` `(`$init_args^ `)`
              `->` type($results) )? $stages attr-dict-with-keyword
```

Op representing a loop consisting of different pipelined stages.
Every stage in the pipeline is a region containing at least one block
argument of type `index` which is the induction variable.
The entry region may additionally have input tensors initialized by
`init_args` if not yet bufferized.
Stages are able to explicitly transfer data from one stage to another
using `pipeline_yield` which are then passed onto the block arguments of
the next stage following the induction variable.

No guarantee is given regarding the order of side effects within a stage
except:
* For a given IV, `Stage[j]` is executed after `Stage[j-1]`.
* For a given stage, IV `i` is executed after IV `i-1`.

Note: Resource allocations performed within a stage may be multiplied by a
lowering to support concurrently running stages.

Produced (in tensor form) by `-snitch-pipeline-copy-compute` from a tiled
`scf.for` loop marked `snitch.dual_buffer`, and expanded (in bufferized/memref
form) by `-snitch-lower-pipeline-op` into an explicit on-ramp/steady-state/
off-ramp `scf.for` nest.

Traits: `InferTypeOpAdaptor`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlockImplicitTerminator<mlir::snitch::PipelineYieldOp>`, `SingleBlock`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lower_bound` | index |
| `upper_bound` | index |
| `step` | index |
| `init_args` | variadic of ranked tensor of any non-token type values |

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of ranked tensor of any non-token type values |

Example (two-stage copy/compute pipeline):

```mlir
%result = snitch.pipeline %c0 to %c8 step %c1 inits(%init) -> tensor<4x4xf32> {
^bb0(%iv0: index):
  %copied = ... // copy stage: prefetch next tile
  snitch.pipeline_yield %copied : tensor<4x4xf32>
}, {
^bb0(%iv1: index, %arg: tensor<4x4xf32>):
  %computed = ... // compute stage: consume previous tile
  snitch.pipeline_yield %computed : tensor<4x4xf32>
}
```

### `snitch.pipeline_yield` (snitch::PipelineYieldOp)

Syntax:

```
operation ::= `snitch.pipeline_yield` ($results `:` type($results)^)? attr-dict
```

Terminator of each `snitch.pipeline` stage; carries the value(s) forwarded to
the next stage's block arguments (or, for the last stage, becomes the op's
results).

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<PipelineOp>`, `ReturnLike`, `Terminator`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `results` | variadic of any non-token type |

Example:

```mlir
snitch.pipeline_yield %computed : tensor<4x4xf32>
```

### `snitch.sync_tensor` (snitch::SyncTensorOp)

Syntax:

```
operation ::= `snitch.sync_tensor` $input `:` type($result) attr-dict
```

Performs synchronization of a tensor returned by a `tensor.microkernel`
operation.
The resulting tensor is guaranteed to consist of the results of any
operations performed by the `tensor.microkernel` operation.

Bufferizes to inserting a `snitch.microkernel_fence`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | ranked tensor of any non-token type values |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | ranked tensor of any non-token type values |

Example:

```mlir
%synced = snitch.sync_tensor %0 : tensor<4x4xf32>
```

### `snitch.tensor.microkernel` (snitch::TensorMicrokernelOp)

Syntax:

```
operation ::= `snitch.tensor.microkernel` (`->` type($results)^ )? $body attr-dict
```

Pre-bufferization version of `memref.microkernel`.
Unlike `memref.microkernel` it is not isolated from above and may also
return tensor operations as outputs via `microkernel_yield`.

Like `memref.microkernel`, operations within the kernel may be executing
asynchronously and cannot be used directly.
A `sync_tensor` operation must be used to make any result tensor of this
operation available.
Failing to do so results in unspecified values within the tensor.

Produced by `-snitch-form-microkernels`, which wraps every
tensor-semantics `linalg` op in one of these.

Traits: `NoRegionArguments`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlock`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `RegionBranchOpInterface`

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of ranked tensor of any non-token type values |

Example:

```mlir
%0:2 = snitch.tensor.microkernel -> tensor<4x4xf32>, tensor<4xf32> {
  %sum, %max = linalg.generic ... -> tensor<4x4xf32>, tensor<4xf32>
  snitch.microkernel_yield %sum, %max : tensor<4x4xf32>, tensor<4xf32>
}
```

## Passes

### `-snitch-form-microkernels`

Wraps every pure-tensor-semantics `linalg.LinalgOp` in a
`snitch.tensor.microkernel`, replacing its uses with a `snitch.sync_tensor`
of the microkernel's result.

### `-snitch-promote-pads-to-l1`

Converts supported `tensor.pad` operations to `start_tensor_transfer` and
`wait_for_tensor_copy` pairs.

Only handles zero-low-pad `tensor.pad` ops whose constant fill value is zero
or "don't care" (`ub.poison`); such pads become
`dma.start_tensor_copy`/`dma.wait_for_tensor_copy` pairs targeting
`#snitch.l1_encoding`.

### `-snitch-promote-operands-to-l1`

Copies tensor operands of `TilingInterface` ops into L1 memory. For every
`TilingInterface` op, each `RankedTensorType` operand gets a
`dma.start_tensor_copy`/`dma.wait_for_tensor_copy` pair inserted and its use
redirected to the wait result; redundant copies are left for CSE to clean up.

### `-snitch-promote-allocs-to-l1`

Marks `bufferization.alloc_tensor` ops (with no `copy` operand) as allocated
in L1 by setting `#snitch.l1_encoding` as their memory space; `alloc_tensor`
ops that do have a `copy` operand are instead rewritten into a
`dma.start_tensor_copy`/`dma.wait_for_tensor_copy` pair into L1.

### `-snitch-lower-l1-allocations`

Lowers `memref.alloca` ops with `#snitch.l1_encoding` memory space into
static offsets/views into a single `snitch.l1_memory_view` byte buffer,
stripping the L1 memory space from the resulting types. Does not support
dynamically sized allocas or non-strided/non-zero-offset memref layouts.

#### Options

```
-l1-memory-bytes : Total size, in bytes, of the L1 scratchpad to allocate into (default 112640).
-assert-compiled : If true, errors if the kernel does not fit into L1 memory. Otherwise, removes the kernel from the output and emits a warning instead.
```

### `-snitch-specialize-dma-code`

Pass performing code specialization for DMA and compute cores while
inserting required synchronization primitives.

Every function originally present in the IR will be cloned and turned into
a "dma" version.
DMA versions have all compute operations (i.e. `memref.microkernel`s)
removed while the original version has all DMA transfer operations removed.
Barriers are inserted where data dependencies require either transfers or
computations to have finished.

The clone's symbol name is recorded on the original function via the
`dma_specialization` discardable attribute.

### `-snitch-eliminate-empty-tensors`

Rewrites `tensor.empty` ops that are anchored on a destination-passing-style
consumer to reuse that consumer's destination tensor instead, ahead of
bufferization.

### `-snitch-bufferize`

Bufferizes the module using upstream One-Shot Bufferize, including
function boundaries (with a static identity layout, since the exported
functions must remain callable with a plain C ABI / xDSL's bare-pointer
calling convention).

When `use-dma-memcpy` is set, buffers are allocated with `memref.alloca`
(suitable for the small, statically sized L1 scratchpad) and copies between
buffers are lowered to `dma.start_transfer`/`dma.wait_for_transfer` pairs
instead of the default `memref.alloc`/`linalg.copy`.

#### Options

```
-use-dma-memcpy : Allocate with memref.alloca and copy via the DMA dialect instead of the upstream defaults.
```

### `-snitch-pipeline-copy-compute`

Finds `scf.for` loops whose tiled compute op is marked with the
`snitch.dual_buffer` unit attribute (attached by the tiling schedule,
e.g. via `transform.annotate`) and lifts them into a two-stage
`snitch.pipeline` (copy stage, compute stage) so a later
`-snitch-lower-pipeline-op` can expand it into a software-pipelined
on-ramp/steady-state/off-ramp loop nest. Operates at the tensor level,
before bufferization.

### `-snitch-lower-forall-op`

Converts a single-induction-variable `scf.forall` (as produced by
thread-level tiling, e.g. `transform.structured.tile_using_forall`) into
a strided `scf.for` where hart `id` processes iterations
`id, id+compute_cores, id+2*compute_cores, ...`, using
`snitch.compute_core_index` to determine `id`. Operates at the memref
level, after bufferization.

#### Options

```
-compute-cores : Number of compute cores (harts) to distribute forall iterations across (default 8).
```

### `-snitch-lower-pipeline-op`

Expands memref-semantics `snitch.pipeline` ops into the concrete
on-ramp/steady-state/off-ramp `scf.for` nest, duplicating any
`memref.alloca`s used by more than one stage (via `scf.index_switch`) so
consecutive iterations of different stages can execute concurrently.
Tensor-semantics `snitch.pipeline` ops (not yet bufferized) are left
untouched. Operates at the memref level, after `-snitch-lower-l1-allocations`.
Only supports statically-sized `memref.alloca` resources.

## Conversion Passes

These are registered centrally in `mlir/include/mlir/Conversion/Passes.td`
rather than in this dialect's own `Transforms/Passes.td`, following upstream
MLIR's convention of keeping dialect-to-LLVM(-adjacent) lowerings under
`Conversion/`.

### `-convert-to-riscv`

_Compile snitch.memref.microkernel bodies to RISC-V assembly via xDSL_

Extracts the body of every `snitch.memref.microkernel` op, hands it to the
`xdsl-opt` tool as a subprocess to be compiled down to Snitch-stream
RISC-V assembly, and replaces the microkernel with a
`snitch.call_microkernel` referencing the produced assembly text.

The subprocess runs the xDSL pipeline
`arith-add-fastmath,convert-linalg-to-memref-stream,test-optimise-memref-stream,test-lower-memref-stream-to-snitch-stream,test-lower-snitch-stream-to-asm`.
On failure, either fails the pass (`assert-compiled=true`) or inlines the
original microkernel body in place and drops the op.

#### Options

```
-xdsl-opt-path   : Path to the 'xdsl-opt' executable to use for kernel compilation.
-assert-compiled : If true, errors if any kernel could not be compiled with xDSL. Otherwise, removes the kernel from the output and emits a warning instead.
```

### `-convert-snitch-to-llvm`

_Convert the snitch dialect to LLVM_

Lowers the remaining `snitch.*` ops (`l1_memory_view`, `barrier`,
`microkernel_fence`, `call_microkernel`, `compute_core_index`) to LLVM,
mostly via calls into `libsnRuntime`'s `snRuntime` (`snrt_cluster_core_idx`,
`snrt_partial_barrier`) or inline RISC-V assembly.

#### Options

```
-barrier-participants : Number of harts participating in each lowered snitch.barrier rendezvous. SpecializeDMACode currently only ever produces a compute-core clone and a DM-core clone synchronizing with each other, so this defaults to 2.
```
