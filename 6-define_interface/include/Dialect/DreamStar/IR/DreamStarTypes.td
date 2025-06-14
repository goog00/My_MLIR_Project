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

// 接口interface 是特性
def DreamStar_TensorType : DreamStar_Type<"DSTensor", "ds_tensor",[ShapedTypeInterface],"::mlir::TensorType"> {

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
          using ::mlir::ShapedType::Trait<DSTensorType>::getElementTypeBitWidth;
          using ::mlir::ShapedType::Trait<DSTensorType>::getRank;
          using ::mlir::ShapedType::Trait<DSTensorType>::getNumElements;
          using ::mlir::ShapedType::Trait<DSTensorType>::isDynamicDim;
          using ::mlir::ShapedType::Trait<DSTensorType>::hasStaticShape;
          using ::mlir::ShapedType::Trait<DSTensorType>::getNumDynamicDims;
          using ::mlir::ShapedType::Trait<DSTensorType>::getDimSize;
          using ::mlir::ShapedType::Trait<DSTensorType>::getDynamicDimIndex;

          ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type type){
              if(shape)
                return mlir::dream_star::DSTensorType::get(getContext(), *shape, type);
              return mlir::dream_star::DSTensorType::get(getContext(), getShape(), type);
          }
          DSTensorType clone(::mlir::ArrayRef<int64_t> shape,int64_t device_id) const {
            return DSTensorType::get(getContext(), shape, getElementType(),device_id);
          }
          DSTensorType clone(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) const {
            return DSTensorType::get(getContext(), shape, elementType,getDeviceId());
          }
          DSTensorType clone(::mlir::ArrayRef<int64_t> shape) const {
            return DSTensorType::get(getContext(), shape, getElementType(),getDeviceId());
          }
          DSTensorType clone(::mlir::Type elementType) const {
            return DSTensorType::get(getContext(), getShape(), elementType,getDeviceId());
          }
          DSTensorType clone() const {
            return DSTensorType::get(getContext(), getShape(), getElementType(),getDeviceId());
          }
      }];

  }



def DreamStar_BufferType : DreamStar_Type<"Buffer", "buffer",[]>{
   let summary = "the summary of dream-star buffer type";

   let description = "description of dream-star buffer type";

   let parameters = (ins 
        ArrayRefParameter<"int64_t">:$devices
   );

   let genStorageClass = 1;

   let hasStorageCustomConstructor = 0;

   let assemblyFormat = "`<`$devices`>`";

   let skipDefaultBuilders = 0;

   let genVerifyDecl = 1;

}


#endif // DIALECT_DREAM_STAR_TYPES_TD