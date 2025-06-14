#ifndef DIALECT_DREAM_STAR_TYPES_TD
#define DIALECT_DREAM_STAR_TYPES_TD
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/DreamStar/IR/DreamStarDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//类型的基类
class DreamStar_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<DreamStar_Dialect, name, traits, baseCppClass> {
  // 设置类型的“助记符”（mnemonic），即该类型在 .mlir 文件中显示的名字。
  // 比如 `!dreamstar.tensor<...>` 中的 "tensor" 就是 mnemonic。
  let mnemonic = typeMnemonic;
  // 定义该类型的完整名称（包括方言名 + 类型助记符）。
  // dialect.name 是 DreamStar_Dialect 的 name 字段（比如 "dreamstar"）
  // 所以最终类型名会是 "dreamstar.tensor" 等。
  let typeName = dialect.name # "." # typeMnemonic;
}


//由于mlir内建的tensor是无法表示数据所处的卡的信息，因此自定义Tensor类型，用来确定所处卡的位置信息，对数据进行数据排布。
def DreamStar_TensorType : DreamStar_Type<"DSTensor", "ds_tensor",[]> {

    // 概述
  let summary = " the summary of dream-star tensor type";

  // 方言的详细描述
  let description = "description of dream-star tensor type";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$device_id
  );

 // 是否生成StorageClass, 无特殊情况，建议设为ture
  let genStorageClass = 1;
    // 不建议改动
  let hasStorageCustomConstructor = 0;

 // 额外的builder 声明
  let builders = [
    TypeBuilder<(ins
      "::mlir::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

  //声明一个序列化的方法,parse（）和 print() 用来显示动态tensor类型。
 let hasCustomAssemblyFormat = 1;

//  let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

 // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

let extraClassDeclaration = [{
    // using TensorType::clone;
    // using ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
    // using ShapedType::Trait<NSTensorType>::getRank;
    // using ShapedType::Trait<NSTensorType>::getNumElements;
    // using ShapedType::Trait<NSTensorType>::isDynamicDim;
    // using ShapedType::Trait<NSTensorType>::hasStaticShape;
    // using ShapedType::Trait<NSTensorType>::getNumDynamicDims;
    // using ShapedType::Trait<NSTensorType>::getDimSize;
    // using ShapedType::Trait<NSTensorType>::getDynamicDimIndex;
    // NSTensorType clone(::mlir::Type elementType) {
    //   return ::llvm::cast<NSTensorType>(cloneWith(getShape(), elementType));
    // }
  }];

}


#endif // DIALECT_DREAM_STAR_TYPES_TD