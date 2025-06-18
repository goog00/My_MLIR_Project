#include "Conversion/Passes.h"
#include "Dialect/DreamStar/Transforms/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pipeline {

void buildBuffeDreamStarBasicPipeline(OpPassManager &pm,
                                      const DreamStarBasicPipelineOptions *options) {
    mlir::dream_star::MarkDistributeParallelParametersPassOptions mark_distribute_parallel_option{
        .DPNums = options.DP_Nums, .TPNums = 1};

        // Dialect/DreamStar/Transforms/Passes.h.inc
    pm.addPass(mlir::dream_star::createMarkDistributeParallelParametersPass(
        mark_distribute_parallel_option));

    pm.addNestedPass<func::FuncOp>(mlir::dream_star::createApplyDistributeTransformPass());

    pm.addNestedPass<func::FuncOp>(mlir::dream_star::createDeviceRegionFusionPass());

    pm.addPass(mlir::dream_star::createConvertDreamStarToLinalgPass());
}

void registerDreamStarBasicPipelines() {
    PassPipelineRegistration<DreamStarBasicPipelineOptions>(
        "dream-star-basic-pipeline", "basic pipeline", buildBufferDreamStarBasicPipeline)
}
}  // namespace mlir::pipeline