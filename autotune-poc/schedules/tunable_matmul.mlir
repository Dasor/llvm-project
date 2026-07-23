// The actual input to driver/autotune.py: a runnable payload (same as
// payload/matmul_1024.mlir) plus a transform schedule whose tile sizes are
// UNRESOLVED transform.tune.knob ops. autotune.py reads this file, finds
// every unresolved knob via its `options = [...]`, and searches over them --
// it never needs to know these happen to be called tile_m/tile_n/tile_k, or
// that there are three of them, or what values they hold; that's all
// extracted straight from this file's text.
//
// Knobs already carrying a `selected` value (none here) would be treated as
// fixed and left untouched by the driver -- useful if you want to explore
// only a subset of a larger schedule's knobs.

func.func @main() {
  %A_init = tensor.empty() : tensor<1024x1024xf32>
  %cA = arith.constant 1.0 : f32
  %A = linalg.fill ins(%cA : f32) outs(%A_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %B_init = tensor.empty() : tensor<1024x1024xf32>
  %cB = arith.constant 2.0 : f32
  %B = linalg.fill ins(%cB : f32) outs(%B_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %C_init = tensor.empty() : tensor<1024x1024xf32>
  %cC = arith.constant 0.0 : f32
  %C = linalg.fill ins(%cC : f32) outs(%C_init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %D = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%C : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %unranked = tensor.cast %D : tensor<1024x1024xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tile_m = transform.tune.knob<"tile_m"> options = [16, 32, 64] -> !transform.param<i64>
    %tile_n = transform.tune.knob<"tile_n"> options = [16, 32, 64] -> !transform.param<i64>
    %tile_k = transform.tune.knob<"tile_k"> options = [16, 32, 64] -> !transform.param<i64>
    %par_tiled, %forall = transform.structured.tile_using_forall %mm num_threads [8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %par_tiled tile_sizes [%tile_m, %tile_n, %tile_k]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled { create_named_contraction } : !transform.any_op
    %contracts = transform.structured.match ops{["vector.contract"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fastmath_contract = transform.param.constant #arith.fastmath<contract> -> !transform.any_param
    transform.annotate %contracts "fastmath" = %fastmath_contract : !transform.any_op, !transform.any_param
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
