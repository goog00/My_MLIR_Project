#include "Conversion/DreamStarToLinalg/DreamStarToLinalg.h"

#include <memory>
#include <optional>

#include "Dialect/DreamStar/IR/DreamStarOps.h"
#include "Dialect/DreamStar/IR/DreamStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace {
struct SoftmaxOpToLinalgPattern final
    : public OpConversionPattern<mlir::dream_star::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult match(dream_star::SoftmaxOp op) const final {
    return llvm::success();
  }

  void rewrite(dream_star::SoftmaxOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto convert = getTypeConverter();
    llvm::SmallVector<Value> out_dy_sizes;
    auto input = adaptor.getInput();
    auto res_type =
        llvm::dyn_cast_or_null<ShapedType>(convert->convertType(op.getType()));
    auto rank = res_type.getRank();
    for (auto i : llvm::index_range(0, rank)) {
      if (!res_type.isDynamicDim(i)) continue;
      auto dim = rewriter.create<tensor::DimOp>(loc, input, i);
      out_dy_sizes.push_back(dim.getResult());
    }
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, res_type.getShape(), res_type.getElementType(), out_dy_sizes);
    auto new_softmax = rewriter.create<linalg::SoftmaxOp>(
        loc, res_type, adaptor.getInput(), output, adaptor.getAxis());
    rewriter.replaceOp(op, new_softmax);
  }
};

struct DeviceKernelOpConvertPattern final
    : public OpConversionPattern<mlir::dream_star::DeviceKernelOp> {

  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(dream_star::DeviceKernelOp op) const final {
    return llvm::success();
  }

  void rewrite(dream_star::DeviceKernelOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    llvm::SmallVector<Type> new_results;
    if (getTypeConverter()
            ->convertTypes(op.getResultTypes(), new_results)
            .failed()) {
      return;
    };

    auto new_op = rewriter.create<dream_star::DeviceKernelOp>(
        loc, new_results, adaptor.getSymName(), adaptor.getDeviceId(),
        adaptor.getArgs());

    rewriter.cloneRegionBefore(op.getRegion(), new_op.getRegion(),
                               new_op.getRegion().end());

    // todo:error: no member named 'getBody' in 'mlir::dream_star::DeviceKernelOp' auto new_block = new_op.getBody();
    // auto new_block = new_op.getBody();

    if (new_op.getRegion().getBlocks().empty()) {
      op->emitError("DeviceKernelOp region has no blocks");
      return;
    }
    auto new_block = &new_op.getRegion().getBlocks().front();

    for (auto [index, arg] : llvm::enumerate(new_block->getArguments())) {
      if (auto ds_tensor =
              llvm::dyn_cast_or_null<dream_star::DSTensorType>(arg.getType())) {
        rewriter.setInsertionPointAfterValue(arg);
        arg.setType(RankedTensorType::get(ds_tensor.getShape(),
                                          ds_tensor.getElementType()));
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            loc, ds_tensor, new_block->getArgument(index));
        rewriter.replaceAllUsesExcept(arg, cast.getResult(0), cast);
      }
    }
    auto return_op = new_block->getTerminator();
    for (auto [index, operand] : llvm::enumerate(return_op->getOperands())) {
      if (auto ds_tensor = llvm::dyn_cast_or_null<dream_star::DSTensorType>(
              operand.getType())) {
        rewriter.setInsertionPointAfterValue(operand);
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            loc, typeConverter->convertType(operand.getType()), operand);

        return_op->setOperand(index, cast.getResult(0));
      }
    }

    for (auto [index, res, new_res] :
         llvm::enumerate(op->getResults(), new_op->getResults())) {
      rewriter.setInsertionPointAfterValue(new_res);
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          loc, res.getType(), new_res);
      rewriter.replaceAllUsesWith(res, cast.getResult(0));
    }

    rewriter.replaceOp(op, new_op);
  };
};
}  // namespace

namespace mlir::dream_star {
namespace {

// 把一个标准 MLIR Tensor（如 RankedTensorType）转换为 NSTensorType，这是
// north_star dialect 自定义的 Tensor 类型。
static Value materializeToDSTensor(OpBuilder &builder, DSTensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  //   检查该输入是 RankedTensorType（MLIR 标准张量类型）。
  assert(isa<RankedTensorType>(inputs[0].getType()));
  // UnrealizedConversionCastOp 是一种 占位符 IR 操作，会在后续 pass 中被转换
  // pass 替换为真正的操作或移除（如果类型等价）。

  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

// 把一个 NSTensorType 转换为标准 TensorType
static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<DSTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

}  // namespace

void initDreamStarToLinalgTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](DSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });

  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });

  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}

void populateDreamStarToLinalgPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  patterns.add<SoftmaxOpToLinalgPattern, DeviceKernelOpConvertPattern>(
      typeConverter, patterns.getContext());
};

}  // namespace mlir::dream_star
