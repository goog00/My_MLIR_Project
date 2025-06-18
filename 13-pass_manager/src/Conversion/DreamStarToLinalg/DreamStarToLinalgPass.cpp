#include <memory>

#include "Conversion/DreamStarToLinalg/DreamStarToLinalg.h"
#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/IR/DreamStarOps.h"
#include "Dialect/DreamStar/IR/DreamStarTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-dream-star-to-linalg"

namespace mlir::dream_star {

#define GEN_PASS_DEF_CONVERTDREAMSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::dream_star

using namespace ::mlir;
using namespace ::mlir::dream_star;

struct DreamStarToLinalgPassPass
    : public mlir::dream_star::impl::ConvertDreamStarToLinalgPassBase<
          DreamStarToLinalgPassPass> {
  void runOnOperation() override;
};

void configDreamStarToLinalgTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferCastOp>();
  target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::dream_star::DSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<DeviceKernelOp>([](DeviceKernelOp op) {
    for (auto type : op.getArgs().getTypes()) {
      if (isa<::mlir::dream_star::DSTensorType>(type)) return false;
    }
    return true;
  });
  // target.addDynamicallyLegalOp<SoftmaxOp>([](Operation* op) {
  //   return !llvm::isa<DeviceKernelOp>(op->getParentOp());
  // });

  target.addDynamicallyLegalOp<SoftmaxOp>(
      [](Operation* op) -> std::optional<bool> {
        return !llvm::isa<DeviceKernelOp>(op->getParentOp());
      });
}

void DreamStarToLinalgPassPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto mode = getOperation();
  TypeConverter type_convert;
  initDreamStarToLinalgTypeConvert(type_convert);
  RewritePatternSet patterns(&getContext());
  populateDreamStarToLinalgPatterns(type_convert, patterns);
  ConversionTarget target(getContext());
  configDreamStarToLinalgTarget(target);
  if (failed(applyPartialConversion(mode, target, std::move(patterns))))

    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
