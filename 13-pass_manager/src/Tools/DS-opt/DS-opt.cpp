#include "Conversion/Passes.h"
#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

//Users/steng/compiler/mlir/my_mlir_project/build/12-operation_lowing_pass
// 1. 实现opt 工具
// 2. 利用opt 运行pass
// '/Users/steng/compiler/mlir/my_mlir_project/build/12-operation_lowing_pass/src/Tools/DS-opt/DS-opt12' '/Users/steng/compiler/mlir/my_mlir_project/12-operation_lowing_pass/test/softmax.mlir' --apply-distribute-transform --mark-distribute-parallel-parameters="DP=5 TP=1"
// 3. 将IR dump 下来 [ir after and tree] && pm option=
// '/Users/steng/compiler/mlir/my_mlir_project/12-operation_lowing_pass/src/Tools/NS-opt/NS-opt10' '/Users/steng/compiler/mlir/my_mlir_project/12-operation_lowing_pass/test/softmax.
// mlir' --mlir-print-ir-after-all --apply-distribute-transform
// 4. debug 选项 debug\debug-only
// '/Users/steng/compiler/mlir/my_mlir_project/12-operation_lowing_pass/src/Tools/NS-opt/NS-opt10' '/Users/steng/compiler/mlir/my_mlir_project/12-operation_lowing_pass/test/softmax.
// mlir' --mark-distribute-parallel-parameters="DP=5 TP=1" --apply-distribute-transform --device-region-fusion --debug
int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;

  registerAllDialects(registry);
  registry.insert<mlir::dream_star::DreamStarDialect>();

  registerAllExtensions(registry);
  mlir::dream_star::registerDreamStarOptPasses();
  mlir::dream_star::registerDreamStarConversionPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "DS modular optimizer driver", registry));
}