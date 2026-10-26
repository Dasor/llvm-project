// Per-hart GEMM microkernel: a thin wrapper around the snitch_cluster
// project's own BLAS kernel, gemm_fp64_opt from
// snitch_cluster/sw/kernels/blas/gemm/src/gemm_fp64.h. Linked by run_gemm.py
// (microkernel=library) in place of the xDSL-generated microkernel assembly.
//
// xDSL's convert-to-riscv still runs and fixes the call signature/symbol name
// (`matmul$xdsl_kernelN`, 3 bare pointers) the rest of the pipeline calls.
// This file exports exactly one such symbol, KERNEL_SYMBOL; run_gemm.py
// compiles it once per symbol it finds in the lowered LLVM-dialect module.
//
// Shapes are this kernel's per-hart tile (the L1 tile, or a peeled
// remainder tile, after the 8-way thread-tiling of M), read by run_gemm.py
// from the microkernel call's memref types and passed in as -D flags:
//   A: TILE_M x TILE_K slice (row stride LDA)
//   B: TILE_K x TILE_N       (row stride LDB)
//   C: TILE_M x TILE_N slice (row stride LDC)
// C is always accumulated into (beta=1); it is copied into L1 (or zeroed)
// once before the K-tile loop, not by this microkernel.
//
// Compiled as C++ with -D__TOOLCHAIN_LLVM__, like upstream: the runtime
// headers use C++ features, and __TOOLCHAIN_LLVM__ selects the
// __builtin_ssr_enable/disable path in ssr.h instead of the named `ssr` CSR,
// which $SNITCH_TOOLCHAIN's assembler doesn't accept.
#include <stdint.h>

#if !defined(TILE_M) || !defined(TILE_N) || !defined(TILE_K) || !defined(KERNEL_SYMBOL)
#error "TILE_M, TILE_N, TILE_K and KERNEL_SYMBOL must be defined (see run_gemm.py)"
#endif
// Row strides of A, B and C (elements). run_gemm.py passes them per kernel:
// e.g. a peeled remainder tile's C is a slice of a wider L1 buffer.
#ifndef LDA
#define LDA TILE_K
#endif
#ifndef LDB
#define LDB TILE_N
#endif
#ifndef LDC
#define LDC TILE_N
#endif

// gvsoc workaround (defined by run_gemm.py only for simulator=gvsoc). gvsoc's
// SSR model stops draining a write stream's FIFO to memory as soon as SSR is
// disabled, and the next stream configuration discards what is left, so a C
// value pushed through ft2 in the last cycles before snrt_ssr_disable() can be
// lost. On RTL the enable bit only affects instructions issued after it, so
// upstream's kernel is correct there. gemm_fp64_opt calls snrt_ssr_disable(),
// which calls __builtin_ssr_disable(); redirecting that builtin lets us wait
// for the write stream to finish without modifying the upstream header.
#ifdef SSR_DRAIN_WORKAROUND
static inline void ssr_disable_after_drain();
#define __builtin_ssr_disable() ssr_disable_after_drain()
#endif

#include "snrt.h"

#ifdef SSR_DRAIN_WORKAROUND
#undef __builtin_ssr_disable
static inline void ssr_disable_after_drain() {
    // SSR status register: bit 31 = done, bit 30 = write lane. Only wait
    // when DM2 is actually configured as a write stream.
    uint32_t status;
    do {
        status = read_ssr_cfg(SNRT_SSR_REG_STATUS, SNRT_SSR_DM2);
    } while ((status >> 30 & 1) && !(status >> 31));
    __builtin_ssr_disable();
}
#endif

// From snitch_cluster/sw/kernels/blas/blas.h, which can't be included as a
// whole: it pulls in every BLAS kernel, including the unpatched gemm_fp64.h
// (see run_gemm.py's patch_upstream_gemm).
static inline double multiply_opt(double multiplicand, double multiplier) {
    if (multiplier)
        return multiplicand * multiplier;
    else
        return 0;
}

#include "gemm_fp64.h"

void gemm_library_kernel(double *A, double *B, double *C) asm(KERNEL_SYMBOL);

// Computes columns [n0, n0 + n) of this hart's C tile.
static inline void gemm_columns(double *A, double *B, double *C, uint32_t n0,
                                uint32_t n) {
    gemm_fp64_opt(/*setup_ssr=*/1, /*partition_banks=*/0, /*transa=*/0,
                  /*transb=*/0, TILE_M, n, TILE_K, A, /*lda=*/LDA, B + n0,
                  /*ldb=*/LDB, /*beta=*/1, C + n0, /*ldc=*/LDC);
}

void gemm_library_kernel(double *A, double *B, double *C) {
#ifdef STAGGER_N
    // All harts stream the same B tile. Starting every hart at a different
    // 8-column block (and wrapping around) spreads their concurrent B reads
    // over different TCDM banks instead of having them chase each other
    // through the same ones.
    const uint32_t blocks = TILE_N / 8;
    const uint32_t n0 = (snrt_cluster_core_idx() % blocks) * 8;
    gemm_columns(A, B, C, n0, TILE_N - n0);
    if (n0) gemm_columns(A, B, C, 0, n0);
#else
    gemm_columns(A, B, C, 0, TILE_N);
#endif
}
