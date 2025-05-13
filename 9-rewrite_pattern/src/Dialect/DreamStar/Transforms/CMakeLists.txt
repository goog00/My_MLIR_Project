add_mlir_dialect_library(MLIRDreamStarTransforms${ch_num}
    ApplyDistributeTransform.cpp
    MarkDistributeParallelParameters.cpp 
    DeviceRegionFusion.cpp

    DEPENDS
    MLIRDreamStarPassesIncGen${ch_num}

    LINK_LIBS PUBLIC 
    MLIRDreamStarDialect${ch_num}
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
)