#ifndef CONVERSION_DREAMSTARTOLINALG_DREAMSTARTOLINALG_H
#define CONVERSION_DREAMSTARTOLINALG_DREAMSTARTOLINALG_H



#include <memory>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
    class TypeConverter;
}

namespace mlir::dream_star {
    void initDreamStarToLinalgTypeConvert(TypeConverter &typeConverter);
    void populateDreamStarToLinalgPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns);

    #define GEN_PASS_DECL_CONVERTDREAMSTARTOLINALGPASS
    #include "Conversion/Passes.h.inc"
} //

#endif