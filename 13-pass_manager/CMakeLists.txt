set(ch_num 13)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
set(TUTORIA_TOOL_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/Tools/DS-opt)
set(TUTORIA_DS_OPT ${TUTORIA_TOOL_DIR}/DS-opt${ch_num})
add_subdirectory(include)
add_subdirectory(src)
add_executable(CH-${ch_num} "main.cpp")
# 将target与定义的方言链接起来
target_link_libraries(CH-${ch_num} 
        MLIRDreamStarDialect${ch_num} 
        MLIRTutorialUtils${ch_num}
        MLIRDreamStarTransforms${ch_num}
        MLIRGPUDialect
        MLIRFuncDialect
        MLIRSCFDialect
        MLIRLinalgDialect
        MLIRParser
        MLIRPass

)

add_subdirectory(test)