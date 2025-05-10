set(LLVM_TARGET_DEFINITIONS DreamStarTypes.td)

## 生成 DreamStar Dialect 的声明
mlir_tablegen(DreamStarDialect.h.inc --gen-dialect-decls -dialect=dream_star)
#生成DreamStar Dialect的实现
mlir_tablegen(DreamStarDialect.cpp.inc --gen-dialect-defs -dialect=dream_star)

mlir_tablegen(DreamStarTypes.h.inc -gen-typedef-decls -dialect=dream_star)
mlir_tablegen(DreamStarTypes.cpp.inc -gen-typedef-defs -dialect=dream_star)

#将生成的命令定义为Target
add_public_tablegen_target(DreamStarDialectIncGen${ch_num})
