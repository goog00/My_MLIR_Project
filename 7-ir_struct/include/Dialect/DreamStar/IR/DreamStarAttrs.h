#ifndef DIALECT_DREAM_STAR_ATTRS_H
#define DIALECT_DREAM_STAR_ATTRS_H
#include "Dialect/DreamStar/IR/DreamStarEnums.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#define GET_ATTRDEF_CLASSES
#include "Dialect/DreamStar/IR/DreamStarAttrs.h.inc"

#endif  // DIALECT_DREAM_STAR_ATTRS_H