add_mlir_dialect_library(MLIRDreamStarDialect${ch_num}
    DreamStarDialect.cpp
    DreamStarTypes.cpp

    DEPENDS
    DreamStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR 
    MLIRTensorDialect
)