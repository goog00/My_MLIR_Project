#ifndef DIALECT_DREAM_STAR_TYPES_H
#define DIALECT_DREAM_STAR_TYPES_H
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#define FIX
#define GET_TYPEDEF_CLASSES
#include "Dialect/DreamStar/IR/DreamStarTypes.h.inc"
#undef FIX

#endif  // DIALECT_DREAM_STAR_TYPES_H