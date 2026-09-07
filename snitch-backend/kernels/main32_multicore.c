#include <stdio.h>
#include <sync_decls.h>
#include <team_decls.h>

#ifndef M
#define M 32
#endif
#ifndef K
#define K 32
#endif
#ifndef N
#define N 32
#endif

// matmul for normal cores
extern double *matmul(double *lhs, double *rhs, double *out);
// matmul for the DM core
extern double *matmul_dma(double *lhs, double *rhs, double *out) asm("matmul$dma");

// Inputs
static double lhs[M * K];
static double rhs[K * N];
static double out[M * N];

int main() {
  if (snrt_is_dm_core()) {
    // The DM core is responsible for initializing the inputs and calling the DMA version of matmul
    for (int i = 0; i < M * K; i++) lhs[i] = (i % 7) + 1;
    for (int i = 0; i < K * N; i++) rhs[i] = (i % 5) + 1;
    matmul_dma(lhs, rhs, out);
  } else {
    // Every compute-capable hart (0-7) now runs the same compiled function
    matmul(lhs, rhs, out);
  }
  snrt_cluster_hw_barrier();  // barrier to ensure all cores have finished before checking results

  if (!snrt_is_dm_core()) return 0; // Only the DM core checks the results and prints the output

  int failed = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++) acc += lhs[i * K + k] * rhs[k * N + j];
      double expected = acc;
      double got = out[i * N + j];
      if (got != expected) {
        printf("mismatch at (%d,%d): got %f expected %f\n", i, j, got,
               expected);
        failed = 1;
      }
    }
  }
  if (!failed) printf("all %d values matched\n", M * N);
  return failed;
}
