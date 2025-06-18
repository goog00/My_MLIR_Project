// RUN: ns-opt %s  --device-region-fusion   --split-input-file | FileCheck %s

// CHECK-LABEL: DreamStar
// CHECK: func.func @main
// CHECK-COUNT-3: dream_star.device_region 
module @DreamStar {
  func.func @main(%arg0: !dream_star.ds_tensor<5x?x?xf32,0>) -> !dream_star.ds_tensor<5x?x?xf32,0> attributes {dp_attr = #dream_star.DP<DP = 3 : 0, 1, 2>, host_func} {
    %0:3 = "dream_star.buffer_cast"(%arg0) <{distribute_attr = #dream_star.DP<DP = 3 : 0, 1, 2>}> : (!dream_star.ds_tensor<5x?x?xf32,0>) -> (!dream_star.ds_tensor<1x?x?xf32,0>, !dream_star.ds_tensor<2x?x?xf32,1>, !dream_star.ds_tensor<2x?x?xf32,2>)
    %1 = "dream_star.softmax"(%0#0) <{axis = 1 : i64}> : (!dream_star.ds_tensor<1x?x?xf32,0>) -> !dream_star.ds_tensor<1x?x?xf32,0>
    %2 = "dream_star.softmax"(%0#1) <{axis = 1 : i64}> : (!dream_star.ds_tensor<2x?x?xf32,1>) -> !dream_star.ds_tensor<2x?x?xf32,1>
    %3 = "dream_star.softmax"(%0#2) <{axis = 1 : i64}> : (!dream_star.ds_tensor<2x?x?xf32,2>) -> !dream_star.ds_tensor<2x?x?xf32,2>
    %4 = "dream_star.buffer_cast"(%1, %2, %3) <{distribute_attr = #dream_star.DP<DP = 3 : 0, 1, 2>}> : (!dream_star.ds_tensor<1x?x?xf32,0>, !dream_star.ds_tensor<2x?x?xf32,1>, !dream_star.ds_tensor<2x?x?xf32,2>) -> !dream_star.ds_tensor<5x?x?xf32,0>
    %5:3 = "dream_star.buffer_cast"(%4) <{distribute_attr = #dream_star.DP<DP = 3 : 0, 1, 2>}> : (!dream_star.ds_tensor<5x?x?xf32,0>) -> (!dream_star.ds_tensor<1x?x?xf32,0>, !dream_star.ds_tensor<2x?x?xf32,1>, !dream_star.ds_tensor<2x?x?xf32,2>)
    %6 = "dream_star.softmax"(%5#0) <{axis = 1 : i64}> : (!dream_star.ds_tensor<1x?x?xf32,0>) -> !dream_star.ds_tensor<1x?x?xf32,0>
    %7 = "dream_star.softmax"(%5#1) <{axis = 1 : i64}> : (!dream_star.ds_tensor<2x?x?xf32,1>) -> !dream_star.ds_tensor<2x?x?xf32,1>
    %8 = "dream_star.softmax"(%5#2) <{axis = 1 : i64}> : (!dream_star.ds_tensor<2x?x?xf32,2>) -> !dream_star.ds_tensor<2x?x?xf32,2>
    %9 = "dream_star.buffer_cast"(%6, %7, %8) <{distribute_attr = #dream_star.DP<DP = 3 : 0, 1, 2>}> : (!dream_star.ds_tensor<1x?x?xf32,0>, !dream_star.ds_tensor<2x?x?xf32,1>, !dream_star.ds_tensor<2x?x?xf32,2>) -> !dream_star.ds_tensor<5x?x?xf32,0>
    return %9 : !dream_star.ds_tensor<5x?x?xf32,0>
  }
}