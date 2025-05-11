#include "Dialect/DreamStar/IR/DreamStarOps.h"

#include <algorithm>

#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/IR/DreamStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Value.h"
#define GET_OP_CLASSES
#include "Dialect/DreamStar/IR/DreamStarOps.cpp.inc"

namespace mlir::dream_star {

void DreamStarDialect::registerOps() {
  llvm::outs() << "register " << getDialectNamespace() << " Op\n";

  addOperations<
#define GET_OP_LIST
#include "Dialect/DreamStar/IR/DreamStarOps.cpp.inc"
      >();
}

::llvm::LogicalResult GetTensorOp::verify() {
  auto device_id = getDeviceId();
  auto buffer = getBuffer();
  if (isa<BlockArgument>(buffer)) {
    auto buffer_type = cast<BufferType>(buffer.getType());
    auto device_ids = buffer_type.getDevices();
    for (auto id : device_ids) {
      if (id == device_id) return llvm::success();
    }
    return llvm::failure();
  }
  auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
  if (!buffer_op) return llvm::failure();
  for (auto tensor : buffer_op.getTensors()) {
    auto tensor_type = cast_or_null<DSTensorType>(tensor.getType());
    if (!tensor_type) return llvm::failure();
    if (device_id == tensor_type.getDeviceId()) {
      if (tensor_type != getType()) return llvm::failure();
      return llvm::success();
    }
  }
  return llvm::failure();
};

::llvm::LogicalResult BufferOp::verify() {
  auto tensors = getTensors();
  auto devices = cast<BufferType>(getType()).getDevices();
  if (tensors.size() == 0) return llvm::failure();
  for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
    auto tensor_type = cast_or_null<DSTensorType>(tensor.getType());
    if (device_id != tensor_type.getDeviceId()) return llvm::failure();
  }
  return llvm::success();
}

::llvm::LogicalResult SoftmaxOp::verify() {
  auto axis = getAxis();
  if (axis < 0) return llvm::failure();
  auto input_type = cast<DSTensorType>(getInput().getType());
  if (axis >= input_type.getShape().size()) return llvm::failure();
  return llvm::success();
}

}  // namespace mlir::dream_star