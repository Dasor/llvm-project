// Host-side driver for run_gemm.py: the DM core initializes A (MxK) and
// B (KxN) and runs the DMA half of the compiled matmul; the 8 compute cores
// run the compute half. Sizes come from -DM/-DN/-DK.
//
// Verification:
//  - gvsoc: run_gemm.py passes -DVERIFY and the DM core checks C against a
//    naive reference (O(MNK), cheap on the ISS).
//  - Verilator: run_gemm.py leaves VERIFY undefined and verify_gemm_ipc.py
//    reads gemm_c out of simulated memory over the testbench's IPC interface
//    after the program exits, then checks it on the host. It looks up
//    gemm_c by name (hence non-static) and recomputes A and B from the init
//    pattern below -- keep the two in sync.
#include <stdio.h>
#include <sync_decls.h>
#include <team_decls.h>

#if !defined(M) || !defined(N) || !defined(K)
#error "M, N and K must be defined (see run_gemm.py)"
#endif

// matmul for normal cores
extern double *matmul(double *lhs, double *rhs, double *out);
// matmul for the DM core
extern double *matmul_dma(double *lhs, double *rhs, double *out) asm("matmul$dma");

// Inputs and output (global so verify_gemm_ipc.py can find them in the ELF)
double gemm_a[M * K];
double gemm_b[K * N];
double gemm_c[M * N];

int main() {
  if (snrt_is_dm_core()) {
    // The DM core is responsible for initializing the inputs and calling the DMA version of matmul
    for (int i = 0; i < M * K; i++) gemm_a[i] = (i % 7) + 1;
    for (int i = 0; i < K * N; i++) gemm_b[i] = (i % 5) + 1;
    matmul_dma(gemm_a, gemm_b, gemm_c);
  } else {
    // Every compute-capable hart (0-7) now runs the same compiled function
    matmul(gemm_a, gemm_b, gemm_c);
  }
  snrt_cluster_hw_barrier();  // barrier to ensure all cores have finished before checking results

  if (!snrt_is_dm_core()) return 0; // Only the DM core checks the results and prints the output

#ifdef VERIFY
  int failed = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++) acc += gemm_a[i * K + k] * gemm_b[k * N + j];
      double expected = acc;
      double got = gemm_c[i * N + j];
      if (got != expected) {
        printf("mismatch at (%d,%d): got %f expected %f\n", i, j, got,
               expected);
        failed = 1;
      }
    }
  }
  if (!failed) printf("all %d values matched\n", M * N);
  return failed;
#else
  printf("gemm done\n");
  return 0;
#endif
}
