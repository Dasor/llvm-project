// Demonstrates the success case: every knob has a `selected` value drawn
// from its `options`. The interpreter then treats each knob exactly like
// `transform.param.constant`, so this schedule is fully deterministic and
// can be run through the same CPU lowering pipeline as
// schedules/baseline_fixed_tiles.mlir. This is a concrete, one-off example
// of the same schedule shape that driver/autotune.py renders for many
// (tile_m, tile_n, tile_k) candidates.
//
// Run:
//   mlir-opt schedules/knobs_resolved_valid.mlir -transform-interpreter \
//     -test-transform-dialect-erase-schedule \
//     -canonicalize -cse \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -buffer-deallocation-pipeline -convert-bufferization-to-memref \
//     -convert-linalg-to-loops \
//     -convert-vector-to-scf="full-unroll" \
//     -lower-vector-mask -lower-vector-multi-reduction \
//     -convert-scf-to-cf -expand-strided-metadata -lower-affine \
//     -convert-arith-to-llvm \
//     -convert-vector-to-llvm="vector-contract-lowering=outerproduct" \
//     -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm \
//     -convert-ub-to-llvm -reconcile-unrealized-casts \
//     -canonicalize -cse \
//   | mlir-runner -e main -entry-point-result=void \
//     -shared-libs=build/lib/libmlir_c_runner_utils.so,build/lib/libmlir_runner_utils.so
//
// Expected: identical output to schedules/baseline_fixed_tiles.mlir -- a
// 256x256 memref where every entry is 512.

func.func @main() {
  %A_init = tensor.empty() : tensor<256x256xf32>
  %cA = arith.constant 1.0 : f32
  %A = linalg.fill ins(%cA : f32) outs(%A_init : tensor<256x256xf32>) -> tensor<256x256xf32>

  %B_init = tensor.empty() : tensor<256x256xf32>
  %cB = arith.constant 2.0 : f32
  %B = linalg.fill ins(%cB : f32) outs(%B_init : tensor<256x256xf32>) -> tensor<256x256xf32>

  %C_init = tensor.empty() : tensor<256x256xf32>
  %cC = arith.constant 0.0 : f32
  %C = linalg.fill ins(%cC : f32) outs(%C_init : tensor<256x256xf32>) -> tensor<256x256xf32>

  %D = linalg.matmul ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
                     outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>

  %unranked = tensor.cast %D : tensor<256x256xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [%tile_m, %tile_n, %tile_k]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
