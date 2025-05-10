#ifndef DIALECT_DREAM_STAR
#define DIALECT_DREAM_STAR
include "mlir/IR/DialectBase.td"

def DreamStar_Dialect : Dialect {

    let name = "dream_star";

    let summary = "summary of DreamStar Dialect";

    let description = "description of Dream Dialect";

    let dependentDialects = ["::mlir::tensor::TensorDialect"];

    let discardableAttrs = (ins);

    let cppNamespace = "::mlir::dream_star";

     // 
    let extraClassDeclaration = [{
        static void sayHello();
    }];

    let hasConstantMaterializer = 0;

    let hasNonDefaultDestructor = 1;

    let hasOperationAttrVerify = 0;

    let hasRegionResultAttrVerify = 0;

    let hasOperationInterfaceFallback = 0;

    let useDefaultAttributePrinterParser = 0;

    let useDefaultTypePrinterParser = 0;

    let hasCanonicalizer = 0;

    let isExtensible = 0;

    let usePropertiesForAttributes = 1;

    

}

#endif // DIALECT_DREAM_STAR