#ifndef DIALECT_DREAM_STAR_OPS_TD
#define DIALECT_DREAM_STAR_OPS_TD

include "Dialect/DreamStar/IR/DreamStarAttrs.td"
include "Dialect/DreamStar/IR/DreamStarConstraints.td"

//class 定义模版
// class —— 模板定义（模板类）
// 类似于 C++ 中的模板/类。
// 不会直接生成实体，而是用于 复用、继承、参数化定义。
// 可以包含 let 字段、继承、默认值、参数等。
// 必须通过 def 或其他 class 派生使用。


//def 定义实例
//  def —— 实体定义（实例）
// 类似于 C++ 中的对象或具体类定义。
// 会被 TableGen 编译为实际的操作、类型、属性、转换等生成代码。
// 用于定义具体的算子、属性、类型等。

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
        OperandType:$input), //声明一个名为 input 的操作数，其类型为 OperandType
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
        //OpBuilder 是用于生成 操作（Op）构造函数 的机制
        //OpBuilder<(ins 参数类型和名称...), [{ 构造逻辑代码 }]>
        //(ins ...)：定义构造该操作所需的参数（C++ 类型 + 名称）。
        //[{}]：C++ 构造代码，用来构造操作数（operands）、属性（attributes）、结果类型（types）等
        //可以写多个 OpBuilder
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