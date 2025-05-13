#ifndef DIALECT_DREAM_STAR_TRANSFORMS_PASSES_H
#define DIALECT_DREAM_STAR_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"

namespace mlir::dream_star {

void populateBufferCastOpCanonicalizationPatterns(RewritePatternSet &patterns);

void populateDeviceRegionFusionPatterns(RewritePatternSet &patterns);

std::unique_ptr<::mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/DreamStar/Transforms/Passes.h.inc"
}  // namespace mlir::dream_star

#endif  // DIALECT_DREAM_STAR_TRANSFORMS_PASSES_H