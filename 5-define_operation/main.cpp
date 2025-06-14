#include <cstddef>

#include "Dialect/DreamStar/IR/DreamStarAttrs.h"
#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/IR/DreamStarOps.h"
#include "Dialect/DreamStar/IR/DreamStarTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

void CH2() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;

  // 初始化上下文环境
  mlir::MLIRContext context(registry);

  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::dream_star::DreamStarDialect>();

  dialect->sayHello();
}

void typeBrief() {
  // 文件定义：llvm-project/mlir/include/mlir/IR/BuiltinTypes.td

  auto context = new mlir::MLIRContext;

  // 浮点数，每种位宽和标准定义一个
  auto f32 = mlir::Float32Type::get(context);
  llvm::outs() << "F32 类型： \t";
  f32.dump();

  auto bf16 = mlir::BFloat16Type::get(context);
  llvm::outs() << "BF16类型：\t";
  bf16.dump();

  // Index 类型，机器相关的整数类型
  auto index = mlir::IndexType::get(context);
  llvm::outs() << "index 类型：\t";
  index.dump();

  // 整数类型，参数：位宽&&有无符号
  auto i32 = mlir::IntegerType::get(context, 32);
  llvm::outs() << "I32 类型： \t";
  i32.dump();

  auto ui16 = mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
  llvm::outs() << "UI16 类型： \t";
  ui16.dump();

  // 张量类型，表示数据，不会有内存布局信息
  auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 张量类型：\t";
  static_tensor.dump();

  // 动态张量
  auto dynamic_tensor =
      mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);

  llvm::outs() << "动态F32 张量类型： \t";

  dynamic_tensor.dump();

  // Memref类型：表示内存
  auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 内存类型：\t";
  basic_memref.dump();

  // 带有布局信息的内存
  auto stride_layout_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32, mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}));
  llvm::outs() << "连续附带布局信息的 F32 内存类型： \t";
  stride_layout_memref.dump();

  // 动态连续附带 affine 布局信息的内存
  auto affine_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32,
      mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}).getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
  affine_memref.dump();

  // 具有内存层级信息的内存
  auto L1_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            1);

  llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
  L1_memref.dump();

  // gpu 私有内存层级的内存
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  auto gpu_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            mlir::gpu::AddressSpaceAttr::get(
                                context, mlir::gpu::AddressSpace::Private));

  llvm::outs() << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型:\t ";
  gpu_memref.dump();

  // 向量类型,定长的一段内存
  auto vector_type = mlir::VectorType::get(3, f32);
  llvm::outs() << "F32 1D向量类型：\t";
  vector_type.dump();

  auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
  llvm::outs() << "f32 2d 向量类型：\t";
  vector_2D_type.dump();
  delete context;
}

void CH3() {
  typeBrief();
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  auto dialect = context.getOrLoadDialect<mlir::dream_star::DreamStarDialect>();
  dialect->sayHello();
  // 静态 DSTensor
  mlir::dream_star::DSTensorType ns_tensor =
      mlir::dream_star::DSTensorType::get(&context, {1, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "Dream Star Tensor 类型 :\t";
  ns_tensor.dump();

  // 动态 DSTensor
  mlir::dream_star::DSTensorType dy_ns_tensor =
      mlir::dream_star::DSTensorType::get(&context,
                                          {mlir::ShapedType::kDynamic, 2, 3},
                                          mlir::Float32Type::get(&context), 3);

  llvm::outs() << "动态 Dream Star Tensor 类型 :\t";
  dy_ns_tensor.dump();
}

void attributeBrief() {
  auto context = new mlir::MLIRContext;
  context->getOrLoadDialect<mlir::dream_star::DreamStarDialect>();

  // Float Attr 表示浮点数的Attribute
  auto f32_attr = mlir::FloatAttr::get(mlir::Float32Type::get(context), 2);
  llvm::outs() << "F32 Attribute: \t";
  f32_attr.dump();

  // Integer Attr : 表示整数的Attribute
  auto i32_attr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), 10);
  llvm::outs() << "I32 Attribute :\t";
  i32_attr.dump();

  // stridelayout attr 表示内存布局信息的Attri
  auto stride_layout_attr = mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1});
  llvm::outs() << "StrideLayout Attribute: \t";
  stride_layout_attr.dump();

  // String Attr 表示字符串的Attribute
  auto str_attr = mlir::StringAttr::get(context, "Hello MLIR");
  llvm::outs() << "String Attribute:\t";
  str_attr.dump();

  // StrRef Attr 表示符号的Attribute
  auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
  llvm::outs() << "SymbolRef Attribute: \t";
  str_ref_attr.dump();

  // Type Attr 储存Type的Attribute
  auto type_attr = mlir::TypeAttr::get(mlir::dream_star::DSTensorType::get(
      context, {1, 2, 3}, mlir::Float32Type::get(context)));
  llvm::outs() << "Type Attribute : \t";
  type_attr.dump();

  // unit attr 一般作为标记使用
  auto unit_attr = mlir::UnitAttr::get(context);
  llvm::outs() << "unit attribute: \t";
  unit_attr.dump();

  auto i64_arr_attr = mlir::DenseI64ArrayAttr::get(context, {1, 2, 3});
  llvm::outs() << "Array Attribute : \t";
  i64_arr_attr.dump();

  auto dense_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({2, 2}, mlir::Float32Type::get(context)),
      llvm::ArrayRef<float>{1, 2, 3, 4});
  llvm::outs() << "Dense Attribute: \t";
  dense_attr.dump();
  delete context;
}

void CH4() {
    attributeBrief();
    // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    auto dialect = context.getOrLoadDialect<mlir::dream_star::DreamStarDialect>();
    // Layout Eunms
    auto nchw = mlir::dream_star::Layout::NCHW;
    llvm::outs() << "NCHW: " << mlir::dream_star::stringifyEnum(nchw) << "\n";
    // LayoutAttr
    auto nchw_attr = mlir::dream_star::LayoutAttr::get(&context, nchw);
    llvm::outs() << "NCHW LayoutAttribute :\t";
    nchw_attr.dump();
    // DataParallelismAttr
    auto dp_attr = mlir::dream_star::DataParallelismAttr::get(&context, 2);
    llvm::outs() << "DataParallelism Attribute :\t";
    dp_attr.dump();
  }


  void CH5() {
    mlir::DialectRegistry registry;

    mlir::MLIRContext context(registry);

    context.getOrLoadDialect<mlir::dream_star::DreamStarDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // module op
    auto module = builder.create<mlir::ModuleOp>(loc, "DreamStar");
    builder.setInsertionPointToStart(module.getBody());

    //const op
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>({2,2});
    auto const_value_1 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
    auto const_value_2 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
    auto tensor_type_1 = mlir::dream_star::DSTensorType::get(&context, shape, f32, 0);
    auto tensor_type_2 = mlir::dream_star::DSTensorType::get(&context, shape, f32, 1);
    auto const_1 = builder.create<mlir::dream_star::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),const_value_2)
    );

    auto const_2 = builder.create<mlir::dream_star::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),const_value_1)
    );

    auto const_3 = builder.create<mlir::dream_star::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32), const_value_2)
    );

    auto const_4 = builder.create<mlir::dream_star::ConstOp>(loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),const_value_2)
    );

    llvm::outs() << "const tensor in device 0: \n";
    const_1->dump();
    llvm::outs() << "const tensor in device 1: \n";
    const_3->dump();

    auto buffer_op = builder.create<mlir::dream_star::BufferOp>(
        loc, mlir::ValueRange({const_1,const_3}));
    
    llvm::outs() << "Buffer Op: \n";
    buffer_op->dump();
    
    auto get_tensor_op_1 = builder.create<mlir::dream_star::GetTensorOp>(loc, tensor_type_1, buffer_op, 0);
    auto get_tensor_op_2 = builder.create<mlir::dream_star::GetTensorOp>(loc, tensor_type_2, buffer_op, 1);

    llvm::outs() << "Get Tensor Op: \n";

    get_tensor_op_1->dump();
    get_tensor_op_2->dump();

    //softmax op
    auto softmax_op = builder.create<mlir::dream_star::SoftmaxOp>(loc, get_tensor_op_1, 1);
    llvm::outs() << "Softmax Op: \n";
    softmax_op->dump();

    // Exp Op
    auto exp_op = builder.create<mlir::dream_star::ExpOp>(loc, get_tensor_op_2);
    llvm::outs() << "Exp Op : \n";
    exp_op->dump();

    // all to all op
    auto out_buffer_op = builder.create<mlir::dream_star::BufferOp>(loc, mlir::ValueRange({const_2, const_4}));
    auto all_to_all_op = builder.create<mlir::dream_star::AllToAllOp>(loc,buffer_op, out_buffer_op);
    llvm::outs() << "all to all op: \n";

    // 将内部 IR（中间表示）打印到标准输出
    all_to_all_op->dump();

  }
  int main() { CH5(); }
