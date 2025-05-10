set(ch_num 3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
add_executable(CH-${ch_num} "main.cpp")
# 将target与定义的方言链接起来
target_link_libraries(CH-${ch_num} MLIRDreamStarDialect${ch_num} MLIRGPUDialect)