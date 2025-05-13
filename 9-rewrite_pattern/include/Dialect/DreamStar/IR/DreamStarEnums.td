#ifndef DIALECT_DREAM_STAR_EUNMS_TD
#define DIALECT_DREAM_STAR_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect/DreamStar/IR/DreamStarTypes.td"

def DreamStar_LAYOUT_NCHW       : I32EnumAttrCase<"NCHW", 0>;
def DreamStar_LAYOUT_NHWC       : I32EnumAttrCase<"NHWC", 1>;
def DreamStar_Layout : I32EnumAttr<"Layout", 
        "Layout of tensor",
        [DreamStar_LAYOUT_NCHW, DreamStar_LAYOUT_NHWC]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::dream_star";
}

def LLH_LayoutAttr : EnumAttr<DreamStar_Dialect, DreamStar_Layout, "Layout">{
    let assemblyFormat = "`<` $value `>`";

    let extraClassDeclaration = [{
        bool isChannelLast();
    }];
}

#endif //DIALECT_DREAM_STAR_EUNMS_TD