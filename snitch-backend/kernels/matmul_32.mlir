func.func @matmul(%a: tensor<32x32xf64>, %b: tensor<32x32xf64>, %out: tensor<32x32xf64>) -> tensor<32x32xf64> {
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf64>, tensor<32x32xf64>)
                          outs(%out : tensor<32x32xf64>) -> tensor<32x32xf64>
  return %result : tensor<32x32xf64>
}
