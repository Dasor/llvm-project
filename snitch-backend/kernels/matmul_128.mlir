func.func @matmul(%a: tensor<128x128xf64>, %b: tensor<128x128xf64>, %out: tensor<128x128xf64>) -> tensor<128x128xf64> {
  %result = linalg.matmul ins(%a, %b : tensor<128x128xf64>, tensor<128x128xf64>)
                          outs(%out : tensor<128x128xf64>) -> tensor<128x128xf64>
  return %result : tensor<128x128xf64>
}
