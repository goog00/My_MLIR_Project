add_mlir_dialect_library(MLIRDreamStarDialect${ch_num}
    DreamStarDialect.cpp
    DreamStarTypes.cpp
    DreamStarAttrs.cpp
    DreamStarOps.cpp

    DEPENDS
    MLIRDreamStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR 
    MLIRTensorDialect
    MLIRDistributeParallelismInterfaces${ch_num}
    MLIRFusionRegionInterfaces${ch_num}
)