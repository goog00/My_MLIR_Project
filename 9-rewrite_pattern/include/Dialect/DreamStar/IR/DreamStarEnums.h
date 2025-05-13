#ifndef DIALECT_DREAM_STAR_EUNMS_H
#define DIALECT_DREAM_STAR_EUNMS_H


#include <cstdint>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"


#define FIX
#include "Dialect/DreamStar/IR/DreamStarEnums.h.inc"
#undef FIX
#endif  // DIALECT_DREAM_STAR_EUNMS_H