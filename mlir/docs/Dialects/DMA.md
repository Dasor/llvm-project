# 'dma' Dialect

[TOC]

The `dma` dialect models asynchronous memory-transfer operations at both the
tensor level (`start_tensor_copy`/`wait_for_tensor_copy`, used pre-
bufferization to stage operands into a target memory space such as
[Snitch's L1 scratchpad](Snitch.md#attributes)) and the memref level
(`start_transfer`/`wait_for_transfer`, used post-bufferization to actually
move bytes). Every transfer-starting op returns a `!dma.token` that a later
`wait_for_*` op consumes to guarantee completion — this async/wait split lets
compute and DMA transfers overlap without committing to a specific hardware
DMA engine's programming model.

The generic `dma.start_transfer`/`dma.wait_for_transfer` pair is
progressively lowered towards actual Snitch hardware in two further steps:
first legalized to the 1D/2D, innermost-contiguous shape Snitch's DMA engine
supports (`-snitch-dma-legalize-dma-operations`, see the
[`snitch_dma` dialect](SnitchDMA.md)), then lowered to calls into
`snRuntime`/inline assembly (`--convert-dma-to-llvm`, below).

## Types

### TokenType

Syntax: `!dma.token`

Type representing a potentially active DMA transfer. An opaque handle
returned by transfer-starting ops; passing it to a `wait_for_*` op blocks
until that transfer (or set of combined transfers) completes.

## Attributes

### CompletedTokenAttr

Syntax: `#dma.completed_token`

Attribute representing an instance of a `!dma.token`
signaling a complete transfer.

Used as the constant value materialized for `dma.completed_token` (the
dialect's `hasConstantMaterializer`), and as the fold result of several ops
when a transfer is trivially a no-op (e.g. `dma.start_transfer` with
identical source/destination, or `dma.wait_for_transfer` on an
already-completed token).

## Operations

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/DMA/IR/DMAOps.td)

### `dma.start_tensor_copy` (dma::StartTensorCopyOp)

Syntax:

```
operation ::= `dma.start_tensor_copy` `of` $copy `to` $memory_space
              ( `pad` `with` (`undef` $undef_padding^) : (`zero`)? `by`
              custom<DynamicIndexList>($high_pad, $static_high_pad)^)?
              custom<TensorCopyTypes>(ref($static_high_pad), type($copy), type($result))
              attr-dict
```

Operation starting a copy of a tensor to another memory space, optionally
adding padding and returning it as a new tensor.
The contained values of the resulting tensor is in an unspecified state.
See `wait_for_tensor_copy` to transform the tensor value into a state
equal to `$copy`.

The operation may optionally add padding at the end of each dimension of
the tensor. Zero is used as the padding value.
The dimensions of the result tensor are computed using
`dims(copy)[i] + high_pad[i]`.

This operation is a noop if `$copy` is already in the given memory space,
no padding is added, and bufferization can elide the copy.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>memory_space</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>static_high_pad</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>undef_padding</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `copy` | ranked tensor of any non-token type values |
| `high_pad` | variadic of index |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | ranked tensor of any non-token type values |
| `token` | `!dma.token` |

Bufferization: if the source buffer is already in the target memory space and
there is no padding, the copy is elided entirely (the result aliases the
source in place). Otherwise a new buffer is allocated — zeroed first via
`dma.start_zero_mem_transfer` when non-`undef` padding was requested — and a
`dma.start_transfer` copies the unpadded region into it.

Examples:

```mlir
// Simple copy, no padding.
%r, %tok = dma.start_tensor_copy of %t to #snitch.l1_encoding
    : tensor<8x16xf32> -> tensor<8x16xf32>

// Copy with static zero padding on both dims.
%r, %tok = dma.start_tensor_copy of %t to #snitch.l1_encoding
    pad with zero by [4, 8] : tensor<8x16xf32> -> tensor<12x24xf32>
```

### `dma.wait_for_tensor_copy` (dma::WaitForTensorCopyOp)

Syntax:

```
operation ::= `dma.wait_for_tensor_copy` `of` $copy `:` type($copy) `to` $transfer_tensor `using` $token `->` type($transfer_tensor) attr-dict
```

Operation asserting that a previous `start_tensor_copy` operation has finished.
Unless `token` is the result of an `completed_token` operation,
`transfer_tensor` and `token` must at runtime be a token and tensor yielded
by a `start_tensor_copy` operation and `copy` the original tensor used in
`start_tensor_copy`.

Once this operation returns, the returned tensor's values are guaranteed
equal to the `copy` operand and in the memory space specified in
`start_tensor_copy`.

Note: The additional `copy` operand is given as it is effectively read by
this operation.
This additionally guarantees that the bufferization frame work does not
perform a write to the underlying buffer of `copy` while the transfer is
in progress.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `BufferizableOpInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `transfer_tensor` | ranked tensor of any non-token type values |
| `token` | `!dma.token` |
| `copy` | ranked tensor of any non-token type values |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | ranked tensor of any non-token type values |

Folds directly to `transfer_tensor` if `token` folds to `#dma.completed_token`.
Must bufferize in place on `transfer_tensor`; lowers to a plain
`dma.wait_for_transfer %token` after buffers are materialized.

Example:

```mlir
%out = dma.wait_for_tensor_copy of %src : tensor<8x16xf32>
    to %r using %tok -> tensor<8x16xf32>
```

### `dma.start_transfer` (dma::StartTransferOp)

Syntax:

```
operation ::= `dma.start_transfer` `from` $source `:` type($source) `to` $dest `:` type($dest) attr-dict
```

Operation performing a DMA transfer from one MemRef to another.
The shapes (including dynamic ones at runtime) of both MemRefs must be
identical with different strides and offsets allowed.

The DMA operation is likely (but not guaranteed) to run asynchronous and
its completion only guaranteed by executing the `wait_for_transfers`
operation with the token returned by this operation.

Due to the unspecified order and concurrency of transfers, the resulting
state of a MemRef is unspecified if at any point two transfers not-yet
completed transfers exist that either write to the same memory location
or writes to a memory location read by another transfer.

Traits: `SameOperandsElementType`, `SameOperandsShape`

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `source` | non-0-ranked memref of any non-token type values |
| `dest` | non-0-ranked memref of any non-token type values |

#### Results:

| Result | Description |
| :----: | ----------- |
| `token` | `!dma.token` |

Folds to `#dma.completed_token` if `source` and `dest` are the same SSA
value.

Example:

```mlir
%tok = dma.start_transfer from %src : memref<64xf32, 1> to %dst : memref<64xf32>
```

### `dma.start_zero_mem_transfer` (dma::StartZeroMemTransferOp)

Syntax:

```
operation ::= `dma.start_zero_mem_transfer` $filled `:` type($filled) attr-dict
```

Starts a DMA transfer which when completed has filled the given MemRef
entirely with bit of 0.
I.e. this is equal to C's `memset` with zero, but asynchronous.

The semantics are identical to a `start_transfer` operation where the
source is a MemRef identical in shape to `filled` consisting of just
0 bits.

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `filled` | non-0-ranked memref of any non-token type values |

#### Results:

| Result | Description |
| :----: | ----------- |
| `token` | `!dma.token` |

Example:

```mlir
%tok = dma.start_zero_mem_transfer %buf : memref<128xf32, 1>
```

### `dma.wait_for_transfer` (dma::WaitForTransferOp)

Syntax:

```
operation ::= `dma.wait_for_transfer` $token attr-dict
```

Operation awaiting for all DMA transfers denoted by its token to have
finished.

Canonicalizes away entirely if `token` folds to a constant
`#dma.completed_token` (waiting on an already-completed token is a no-op).

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `token` | `!dma.token` |

Example:

```mlir
dma.wait_for_transfer %tok
```

### `dma.completed_token` (dma::CompletedTokenOp)

Syntax:

```
operation ::= `dma.completed_token` attr-dict
```

Op returning a special value representing a completed DMA transfer.
Passing this token to `wait_for_transfers` will always return immediately.

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| :----: | ----------- |
| `token` | `!dma.token` |

Folds to `#dma.completed_token`. Lowered by `--convert-dma-to-llvm` to
`llvm.mlir.constant 0 : i32`.

Example:

```mlir
%tok = dma.completed_token
```

### `dma.combine_tokens` (dma::CombineTokensOp)

Syntax:

```
operation ::= `dma.combine_tokens` $tokens attr-dict
```

Op combining multiple DMA tokens into one.
Awaiting the token returned by this function is equal in effect as if each
token was awaited independently in unspecified order.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `tokens` | variadic of `!dma.token` |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | `!dma.token` |

Lowered by `--convert-dma-to-llvm` to a chain of `llvm.umax` over the token
values, under a single-channel-DMA assumption (a higher completed-transfer
id implies all lower ids are also complete).

Example:

```mlir
%combined = dma.combine_tokens %tok0, %tok1, %tok2
```

## Extensions

### `DMACoreSpecializationOpInterfaceImpl`

`mlir/include/mlir/Dialect/DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.h`

Registers, via a `DialectRegistry` extension keyed on `DMADialect`, external
models attaching [`snitch`'s core-specialization
interfaces](Snitch.md#interfaces) to the `dma` dialect's transfer ops, so
`-snitch-specialize-dma-code` knows how to handle them when cloning a
function for the DMA core vs. the compute core:

| Op | `replaceWithNoop` | `needsSynchronization` |
|---|---|---|
| `dma.start_transfer` | replaced with `dma.completed_token` | `false` |
| `dma.start_zero_mem_transfer` | replaced with `dma.completed_token` | `false` |
| `dma.wait_for_transfer` | op erased | `true` |

I.e. on a core specialization that doesn't run DMA ops, transfer-starting
ops degrade to a trivial completed token (so downstream waits still fold
away for free), and a bare wait is simply dropped — while `wait_for_transfer`
is marked as needing cross-core synchronization wherever it does survive.

Note: `StartTensorCopyOp`/`WaitForTensorCopyOp`'s `BufferizableOpInterface`
implementations are *not* part of this extension — they are declared inline
in `DMAOps.td` directly, since bufferization is core to those ops' semantics
rather than an optional add-on.

## Conversion Passes

Registered centrally in `mlir/include/mlir/Conversion/Passes.td`.

### `-convert-dma-to-llvm`

_Convert the dma and snitch_dma dialects to LLVM_

Lowers `dma.*` ops (and, in the same pass, [`snitch_dma.stat`](SnitchDMA.md))
to LLVM, using a 32-bit index bitwidth throughout since Snitch is a fixed
RV32 target:

- `!dma.token` → `i32`.
- `dma.start_transfer` (rank 1 or 2 only) → a call to `snrt_dma_start_1d`/
  `snrt_dma_start_2d` (external `snRuntime` functions), computing element
  size, inner size, and strides from the memref descriptor.
- `dma.start_zero_mem_transfer` → for a fully contiguous destination,
  repeated DMA copies from a fixed hardware zero-region
  (`0x10030000`, size `0x10000`); otherwise an `scf.for` loop nest over
  non-contiguous outer dimensions, recursing per contiguous slice and
  combining the resulting tokens.
- `dma.wait_for_transfer` → a spin-loop repeatedly calling `snitch_dma.stat`
  and comparing against the awaited token until it is reached.
- `dma.completed_token` → `llvm.mlir.constant 0 : i32`.
- `dma.combine_tokens` → a chain of `llvm.umax`.
- `snitch_dma.stat` → an `llvm.inline_asm` emitting the Snitch `dmstati`
  instruction (opcode `0x2b`, func3 `0`, func7 `0b100`).

No options.
