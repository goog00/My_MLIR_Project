#ifndef DIALECT_DREAM_STAR_CONSTRAINTS_TD
#define DIALECT_DREAM_STAR_CONSTRAINTS_TD
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

// 定义约束条件（Constraints），用于检查操作（Op）中的输入、输出、属性等是否满足预期类型或结构
// 自定义谓词（CPred）

def AnyDSTensor : Type<And<[CPred<"::mlir::isa<::mlir::dream_star::DSTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::dream_star::DSTensorType>($_self).getShape().size() > 0">]>>;


def AnyBuffer  : Type<And<[CPred<"::mlir::isa<::mlir::dream_star::BufferType>($_self)">,
                           CPred<"::mlir::cast<::mlir::dream_star::BufferType>($_self).getDevices().size() > 0">]>>;


def DSTensorOrBuffer : AnyTypeOf<[AnyDSTensor,AnyBuffer]>;

#endif // DIALECT_DREAM_STAR_CONSTRAINTS_TD 