#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::dream_star {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
}  // namespace mlir::dream_star

#endif