
#include "Dialect/DreamStar/IR/DreamStarDialect.h"

#include "llvm/Support/raw_ostream.h"
#define FIX
#include "Dialect/DreamStar/IR/DreamStarDialect.cpp.inc"
#undef FIX

namespace mlir::dream_star {

void DreamStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";
}

DreamStarDialect::~DreamStarDialect() {
  llvm::outs() << "destorying  " << getDialectNamespace() << "\n";
}

void DreamStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}
}  // namespace mlir::dream_star