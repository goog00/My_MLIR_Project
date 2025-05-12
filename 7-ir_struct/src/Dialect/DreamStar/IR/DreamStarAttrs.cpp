#include "Dialect/DreamStar/IR/DreamStarAttrs.h"

#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/IR/DreamStarEnums.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#define FIX
#define GET_ATTRDEF_CLASSES
#include "Dialect/DreamStar/IR/DreamStarAttrs.cpp.inc"
#include "Dialect/DreamStar/IR/DreamStarEnums.cpp.inc"

namespace mlir::dream_star {
    
void DreamStarDialect::registerAttrs() {
  llvm::outs() << "Register " << getDialectNamespace() << " Attr\n";

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/DreamStar/IR/DreamStarAttrs.cpp.inc"
      >();
}

bool LayoutAttr::isChannelLast() { return getValue() == Layout::NHWC; }


}  // namespace mlir::dream_star

#undef FIX