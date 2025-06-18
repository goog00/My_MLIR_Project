#include "Dialect/DreamStar/IR/DreamStarDialect.h"
#include "Dialect/DreamStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::dream_star {
#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Dialect/DreamStar/Transforms/Passes.h.inc"
}  // namespace mlir::dream_star

using namespace ::mlir;
using namespace ::mlir::dream_star;

struct ApplyDistributeTransformPass
    : ::mlir::dream_star::impl::ApplyDistributeTransformPassBase<ApplyDistributeTransformPass> {
    using ApplyDistributeTransformPassBase<
        ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;
    void runOnOperation() override;
};

void ApplyDistributeTransformPass::runOnOperation() {
    llvm::outs() << "run in: " << getPassName() << "\n";
    auto func = getOperation();
    llvm::outs() << "root op: " << func->getName() << "\n";

    auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(func->getAttr(KDPAttrName));
    if (!dp_attr) llvm_unreachable("error!");
    func->walk([&](mlir::Operation* op) {
        if (auto dis_op = llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
            if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
                llvm::outs() << "Apply DataParallelism to " << op->getName() << "\n";
                // 原始操作被擦除
                op->erase();
            };
        }
    });
    llvm::outs() << "run out: " << getPassName() << "\n\n";
}

std::unique_ptr<::mlir::Pass> mlir::dream_star::createApplyDistributeTransformPass() {
    return std::make_unique<ApplyDistributeTransformPass>();
}