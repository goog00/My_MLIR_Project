#ifndef DIALECT_DREAM_STAR_TRANSFORMS_PASSES_TD
#define DIALECT_DREAM_STAR_TRANSFORMS_PASSES_TD
include "mlir/Pass/PassBase.td"


def MarkDistributeParallelParametersPass : Pass<"mark-distribute-parallel-parameters","::mlir::ModuleOp"> {
    let summary = "标记全局并行参数";
    let description = [{
       "标记全局并行参数";
    }];

    let dependentDialects = [
        "::mlir::dream_star::DreamStarDialect",
        "::mlir::tensor::TensorDialect"
    ];
    let options = [
        Option<"DPNums", "DP", "std::int64_t", "1", "DPNums dec">,
        Option<"TPNums", "TP", "std::int64_t", "1", "TPNums dec">
    ];

    let statistics = [
        Statistic<"EPNums", "ep-nums", "Number of EP">
    ];

}

def ApplyDistributeTransformPass : Pass<"apply-distribute-transform","::mlir::func::FuncOp"> {
    let summary = "根据标记的并行参数进行变换";
    let description = [{
        "根据标记的并行参数进行变换。"
    }];
    let dependentDialects = [
        "::mlir::dream_star::DreamStarDialect",
        "::mlir::tensor::TensorDialect"
    ];
    let constructor = "mlir::dream_star::createApplyDistributeTransformPass()";
}

def DeviceRegionFusionPass : Pass<"device-region-fusion","::mlir::func::FuncOp">{
    let summary = "相同device的计算合并";
    let description = [{
        "相同device的计算合并"
    }];
    let dependentDialects = [
        "::mlir::dream_star::DreamStarDialect",
        "::mlir::tensor::TensorDialect"
    ];

}



#endif  // DIALECT_DREAM_STAR_TRANSFORMS_PASSES_TD
