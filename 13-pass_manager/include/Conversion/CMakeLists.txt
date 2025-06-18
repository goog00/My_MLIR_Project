set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DreamStarConversion)
add_public_tablegen_target(MLIRDreamStarConversionPassesIncGen${ch_num})
add_dependencies(mlir-headers MLIRDreamStarConversionPassesIncGen${ch_num})