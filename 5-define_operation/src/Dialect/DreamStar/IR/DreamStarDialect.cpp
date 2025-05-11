
#include "Dialect/DreamStar/IR/DreamStarDialect.h"

#include "llvm/Support/raw_ostream.h"
#define FIX
#include "Dialect/DreamStar/IR/DreamStarDialect.cpp.inc"
#undef FIX

namespace mlir::dream_star {

void DreamStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";
  //没有下面的代码就异常：LLVM ERROR: can't create type 'mlir::dream_star::DSTensorType' because storage uniquer isn't initialized: the dialect was likely not loaded, or the type wasn't added with addTypes<...>() in the Dialect::initialize() method.
  registerType();
  registerAttrs();
  registerOps();
}

DreamStarDialect::~DreamStarDialect() {
  llvm::outs() << "destorying  " << getDialectNamespace() << "\n";
}

void DreamStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}
}  // namespace mlir::dream_star