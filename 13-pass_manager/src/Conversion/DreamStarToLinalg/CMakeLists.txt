add_mlir_conversion_library(MLIRDreamStarToLinalg${ch_num}

    DreamStarToLinalg.cpp
    DreamStarToLinalgPass.cpp 

    DEPENDS
    MLIRDreamStarConversionPassesIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRDreamStarDialect${ch_num}
    MLIRPass
    MLIRTransformUtils
)