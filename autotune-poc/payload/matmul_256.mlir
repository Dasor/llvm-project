// Standalone linalg.matmul payload IR: 256x256x256, f32.
//
// A is filled with 1.0, B is filled with 2.0, C starts at 0.0, so every
// output entry is deterministically K * 1.0 * 2.0 = 512.0. This makes
// correctness trivial to check from the printed output: every value must
// equal 512.

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
