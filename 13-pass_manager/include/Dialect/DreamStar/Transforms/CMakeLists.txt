set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DreamStarOpt)
add_public_tablegen_target(MLIRDreamStarPassesIncGen${ch_num})
add_dependencies(mlir-headers MLIRDreamStarPassesIncGen${ch_num})