#ifndef PIPELINES_PIPELINS_H
#define PIPELINES_PIPELINS_H
#include <cstdint>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::pipeline {
struct DreamStarBasicPipelineOptions : public PassPipelineOptions<DreamStarBasicPipelineOptions> {
    PassOptions::Option<int64_t> DP_Nums{*this, "DP_Nums", llvm::cl::desc("数据并行参数"),
                                         llvm::cl::init(1)};
};

void buildDreamStarBasicPipeline(OpPassManager &pm, const DreamStarBasicPipelineOptions &options);

void registerDreamStarBasicPipelines();

}  // namespace mlir::pipeline

#endif