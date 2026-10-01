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
// Shapes come from the L1 tile sizes and the 8-way thread-tiling of the M
// dimension, passed in as -D flags by run_gemm.py:
//   A: TILE_M x TILE_K slice (row stride TILE_K)
//   B: TILE_K x TILE_N       (row stride TILE_N)
//   C: TILE_M x TILE_N slice (row stride TILE_N)
// C is always accumulated into (beta=1); it is zeroed once via a separate DMA
// transfer before the K-tile loop, not by this microkernel.
//
// Compiled as C++ with -D__TOOLCHAIN_LLVM__, like upstream: the runtime
// headers use C++ features, and __TOOLCHAIN_LLVM__ selects the
// __builtin_ssr_enable/disable path in ssr.h instead of the named `ssr` CSR,
// which $SNITCH_TOOLCHAIN's assembler doesn't accept.
#include <stdint.h>

#if !defined(TILE_M) || !defined(TILE_N) || !defined(TILE_K) || !defined(KERNEL_SYMBOL)
#error "TILE_M, TILE_N, TILE_K and KERNEL_SYMBOL must be defined (see run_gemm.py)"
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

void gemm_library_kernel(double *A, double *B, double *C) {
    gemm_fp64_opt(/*setup_ssr=*/1, /*partition_banks=*/0, /*transa=*/0,
                  /*transb=*/0, TILE_M, TILE_N, TILE_K, A, /*lda=*/TILE_K, B,
                  /*ldb=*/TILE_N, /*beta=*/1, C, /*ldc=*/TILE_N);
}
