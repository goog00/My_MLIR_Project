
include "mlir/Pass/PassBase.td"

def ConvertDreamStarToLinalgPass : Pass<"convert-dream-star-to-linalg","::mlir::ModuleOp"> {
    let summary = "标记全局并行参数";
    let description = [{
    "标记全局并行参数"
    }];
    let dependentDialects = [
        "::mlir::dream_star::DreamStarDialect",
        "::mlir::tensor::TensorDialect",
        "::mlir::linalg::LinalgDialect"
    ];

}