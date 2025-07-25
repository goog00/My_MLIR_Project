set(LIBS 

  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

   # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in mlir-opt.cpp.
  MLIRToLLVMIRTranslationRegistration
  MLIRDreamStarDialect${ch_num}
  MLIRDreamStarTransforms${ch_num}
  MLIRTutorialUtils${ch_num}
  MLIRDreamStarToLinalg${ch_num}

)

add_mlir_tool(DS-opt${ch_num}
  DS-opt.cpp

  DEPENDS 
  ${LIBS}
)

target_link_libraries("DS-opt${ch_num}" PRIVATE ${LIBS})
llvm_update_compile_flags(DS-opt${ch_num})

mlir_check_all_link_libraries(DS-opt${ch_num})
export_executable_symbols_for_plugins(DS-opt${ch_num})