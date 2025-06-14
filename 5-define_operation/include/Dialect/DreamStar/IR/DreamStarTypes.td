#ifndef DIALECT_DREAM_STAR_TYPES_TD
#define DIALECT_DREAM_STAR_TYPES_TD
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/DreamStar/IR/DreamStarDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"


class DreamStar_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<DreamStar_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = dialect.name # "." # typeMnemonic;
}


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

  //声明一个序列化的方法
 let hasCustomAssemblyFormat = 1;

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



def DreamStar_BufferType : DreamStar_Type<"Buffer", "buffer",[]>{
   let summary = "the summary of dream-star buffer type";

   let description = "description of dream-star buffer type";

   let parameters = (ins 
        ArrayRefParameter<"int64_t">:$devices
   );
    //当你设置 genStorageClass = 1，TableGen 会自动生成一个名为 BufferTypeStorage 的类，
    //里面包含你定义的参数 devices（类型是 ArrayRef<int64_t>）的存储和访问接口。
   let genStorageClass = 1;

   let hasStorageCustomConstructor = 0;

   let assemblyFormat = "`<`$devices`>`";

   let skipDefaultBuilders = 0;

   let genVerifyDecl = 1;

}


#endif // DIALECT_DREAM_STAR_TYPES_TD