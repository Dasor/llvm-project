# 'snitch_dma' Dialect

[TOC]

Dialect dealing with all implementation details specific to Snitch's DMA
engine.
Used to progressively lower and optimize the [`dma` dialect](DMA.md).

Snitch's DMA engine only supports 1D or 2D transfers where at most the
innermost dimension is non-unit-strided (contiguous); the outer dimension may
have arbitrary stride. The `-snitch-dma-legalize-dma-operations` pass below
normalizes arbitrary-rank/arbitrary-stride `dma.start_transfer` ops down to
that shape before they ever reach this dialect's one op, `snitch_dma.stat`,
and the final LLVM lowering in [`--convert-dma-to-llvm`](DMA.md#conversion-passes).

## Operations

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SnitchDMA/IR/SnitchDMAOps.td)

### `snitch_dma.stat` (snitch_dma::StatOp)

Syntax:

```
operation ::= `snitch_dma.stat` attr-dict
```

Returns the id of the last DMA transfer that has been completed. Models
Snitch's `dmstati` status-read instruction at the dialect level; used by
[`dma.wait_for_transfer`'s LLVM lowering](DMA.md#conversion-passes) to poll
for transfer completion.

Interfaces: `InferTypeOpInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on mlir::snitch_dma::QueueResource}`

The read effect is on a custom side-effect `Resource` (`QueueResource`,
named `"queue"`) representing DMA-queue state, rather than a memory
location — this prevents the op from being CSE'd/hoisted across other DMA
ops that mutate queue state.

#### Results:

| Result | Description |
| :----: | ----------- |
| `completed_id` | 32-bit signless integer |

Example:

```mlir
%last_done = snitch_dma.stat
%not_done = arith.cmpi ult, %last_done, %token : i32
```

## Passes

### `-snitch-dma-legalize-dma-operations`

Rewrites arbitrary-rank/arbitrary-stride `dma.start_transfer` ops (from the
[`dma` dialect](DMA.md)) into the 1D/2D, at-most-outer-dimension-strided form
Snitch's DMA hardware supports, using a dynamic legality check
(`rank <= 2` and at most the outermost dimension non-contiguous) and three
rewrite patterns:

- **Rank reduction** — drops a leading unit-size outer dimension via
  `memref.subview`.
- **Collapse** — collapses contiguous inner dimensions into one
  (`memref.collapse_shape`); if more than one non-contiguous outer dimension
  remains, emits an `scf.for` loop nest that slices the memref per outer
  index, issues one `dma.start_transfer` per iteration, and merges the
  resulting tokens with `dma.combine_tokens`.
- **Expand** — adds a synthetic contiguous inner unit dimension
  (`memref.expand_shape`) when no contiguous dimension exists at all.

No options.
