
module @DreamStar {
  func.func @main(%arg0: !dream_star.ds_tensor<5x?x?xf32,0>) -> !dream_star.ds_tensor<5x?x?xf32,0> attributes {dp_attr = #dream_star.DP<DP = 2 : 0, 1>, host_func} {
    %0 = "dream_star.softmax"(%arg0) <{axis = 1 : i64}> : (!dream_star.ds_tensor<5x?x?xf32,0>) -> !dream_star.ds_tensor<5x?x?xf32,0>
    %1 = "dream_star.softmax"(%0) <{axis = 1 : i64}> : (!dream_star.ds_tensor<5x?x?xf32,0>) -> !dream_star.ds_tensor<5x?x?xf32,0>
    return %1 : !dream_star.ds_tensor<5x?x?xf32,0>
  }
}