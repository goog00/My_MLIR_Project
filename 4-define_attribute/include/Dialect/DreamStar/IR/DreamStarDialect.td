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
        void registerType();
        void registerAttrs();
    }];


    let hasConstantMaterializer = 0;

    // 是否生成默认的析构函数
    let hasNonDefaultDestructor = 1;

    let hasOperationAttrVerify = 0;

    let hasRegionResultAttrVerify = 0;

    let hasOperationInterfaceFallback = 0;

    // MLIR默认的属性解析输出 If 0 ==》.NCHW LayoutAttribute :  dialect has no registered attribute printing hook
    let useDefaultAttributePrinterParser = 1;
    // 使用MLIR默认的类型解析输出.
    let useDefaultTypePrinterParser = 1;

    let hasCanonicalizer = 0;

    let isExtensible = 0;

    let usePropertiesForAttributes = 1;

    

}

#endif // DIALECT_DREAM_STAR