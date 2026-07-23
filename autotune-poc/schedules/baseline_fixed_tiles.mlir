// Step 2: payload + a transform-dialect schedule that tiles linalg.matmul
// with LITERAL tile sizes (no knobs yet). This establishes the working
// tiling + full CPU lowering pipeline before knobs are introduced in
// step 3. Run via driver/run_one.sh or the pipeline documented in the
// top-level README.md.

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
    // m, n, k tile sizes -- fixed for now, will become knobs in step 3.
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %mm tile_sizes [32, 32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
