#include <stdio.h>
#include <sync_decls.h>
#include <team_decls.h>

#define M 32
#define K 32
#define N 32

// matmul for normal cores
extern double *matmul(double *lhs, double *rhs, double *out);
// matmul for the DM core
extern double *matmul_dma(double *lhs, double *rhs, double *out) asm("matmul$dma");

// function to read the mcycle CSR for timing
static inline uint32_t snitch_tune_mcycle(void) {
  uint32_t r;
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  return r;
}

// Inputs
static double lhs[M * K];
static double rhs[K * N];
static double out[M * N];

int main() {
  // Not cycle-accurate (gvsoc is a non-cycle-accurate ISS but deterministic in simulated work
  // which is all the autotuner needs
  uint32_t t0 = 0;
  if (snrt_is_dm_core()) {
    // The DM core is responsible for initializing the inputs and calling the DMA version of matmul
    for (int i = 0; i < M * K; i++) lhs[i] = (i % 7) + 1;
    for (int i = 0; i < K * N; i++) rhs[i] = (i % 5) + 1;
    t0 = snitch_tune_mcycle();
    matmul_dma(lhs, rhs, out);
  } else {
    // Every compute-capable hart (0-7) now runs the same compiled function
    matmul(lhs, rhs, out);
  }
  snrt_cluster_hw_barrier();  // barrier to ensure all cores have finished before checking results

  if (!snrt_is_dm_core()) return 0; // Only the DM core checks the results and prints the output

  uint32_t total_cycles = snitch_tune_mcycle() - t0;

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
  printf("total_cycles: %u\n", total_cycles);
  return failed;
}
