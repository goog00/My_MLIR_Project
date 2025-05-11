#ifndef DIALECT_DREAM_STAR_OPS_TD
#define DIALECT_DREAM_STAR_OPS_TD

include "Dialect/DreamStar/IR/DreamStarAttrs.td"
include "Dialect/DreamStar/IR/DreamStarConstraints.td"


class DreamStar_Op<string mnemonic, list<Trait> traits = []> 
        : Op<DreamStar_Dialect, mnemonic, traits> {
    let summary = cppNamespace#opName#"op";

    let description = "$_name op";        
}

// 计算图中常见的单目OP
class DreamStar_UnaryOp<string mnemonic, 
                        Type OperandType, 
                        Type resultType = OperandType, 
                        list<Trait> traits = [], dag attributes = (ins)>
                        : DreamStar_Op<mnemonic, traits#[]>{
     // !con 拼接作用                      
    let arguments = !con((ins 
        OperandType:$input),
        attributes
    );

    let results = (outs 
        resultType:$result);
}

// 计算图中常见的双目OP
class DreamStar_BinaryOp<string mnemonic, 
                        Type OperandType, 
                        Type resultType = OperandType, 
                        list<Trait> traits = [], dag attributes = (ins)>
                        : DreamStar_Op<mnemonic, traits#[]>{
     // !con 拼接作用                      
    let arguments = !con((ins 
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
    );

    let results = (outs 
        resultType:$result);
}


def DreamStar_ConstOp : DreamStar_Op<"const",[]> {
    let arguments = (ins ElementsAttr:$value);

    let results = (outs AnyDSTensor:$result);
}

def DreamStar_SoftmaxOp : DreamStar_UnaryOp<"softmax", AnyDSTensor, AnyDSTensor, [], (ins I64Attr:$axis)>{
    //需要实现一个检测函数
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis),
          [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
          }]
        >
    ];
}

def DreamStar_ExpOp : DreamStar_UnaryOp<"exp",AnyDSTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input),
        [{
           $_state.addOperands(input);
           $_state.addTypes(input.getType());
        }]>
    ];
}

def DreamStar_AddOp : DreamStar_BinaryOp<"add", AnyDSTensor>;
def DreamStar_SubOp : DreamStar_BinaryOp<"sub", AnyDSTensor>;
def DreamStar_MulOp : DreamStar_BinaryOp<"mul", AnyDSTensor>;
def DreamStar_DivOp : DreamStar_BinaryOp<"div", AnyDSTensor>;

def DreamStar_AllToAllOp : DreamStar_Op<"all_to_all",[]>{
    let arguments = (ins 
        AnyBuffer:$input,
        AnyDSTensor:$output
    );
}

def DreamStar_BufferOp : DreamStar_Op<"buffer",[]>{
    let description = "将多个device_id的tensor 组合一个buffer";
    let arguments = (ins 
        Variadic<AnyDSTensor>:$tensors
    );

    let results = (outs
        AnyBuffer:$result
    );

    // DreamStarOps::BufferOp::verify() 
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors),
            [{
                $_state.addOperands(tensors);
                ::llvm::SmallVector<int64_t> devices;
                for (auto tensor : tensors) {
                    auto tensor_type = 
                         llvm::cast<::mlir::dream_star::DSTensorType>(tensor.getType());
                    devices.push_back(tensor_type.getDeviceId());    
                }
                $_state.addTypes(::mlir::dream_star::BufferType::get($_state.getContext(), devices));
            }]>
    ];
}

def DreamStar_GetTensorOp : DreamStar_Op<"get_tensor",[]>{
    let description = "从buffer中取出指定device_id的tensor";
    let arguments = (ins 
        AnyBuffer:$buffer,
        I64Attr:$device_id
    );

    let results = (outs
        AnyDSTensor:$result
    );

    //DreamStarOps::GetTensorOp::verify()
    let hasVerifier = 1;
}

def DreamStar_PrintOp : DreamStar_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}


#endif // DIALECT_DREAM_STAR_OPS_TD