add_mlir_dialect_library(MLIRDreamStarDialect${ch_num}
    DreamStarDialect.cpp

    ADDITIONAL_HEADER_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Dialect/DreamStar

    DEPENDS
    DreamStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR 
    MLIRTensorDialect
)