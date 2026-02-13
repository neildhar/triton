#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-infer-layout-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gluon {

namespace {

//===----------------------------------------------------------------------===//
// LayoutLattice - Lattice value for layout inference
//===----------------------------------------------------------------------===//

/// The layout lattice represents the inferred encoding for a value.
/// States (ordered from bottom to top):
///   - Uninitialized: No encoding has been set yet
///   - Soft(encoding): An encoding that can be overridden (from ops like
///                     reshape, join, split, etc.)
///   - Hard(encoding): A fixed encoding that cannot be overridden
///   - Overdefined: Conflicting hard encodings were found
struct LayoutLattice : public dataflow::AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutLattice)
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override {
    if (isOverdefined) {
      os << "overdefined";
    } else if (encoding) {
      os << (mayVary ? "soft(" : "hard(") << encoding << ")";
    } else {
      os << "uninitialized";
    }
  }

  /// Returns true if this lattice has a valid encoding (not uninitialized or
  /// overdefined).
  bool hasEncoding() const { return encoding && !isOverdefined; }

  /// Set to a new encoding. Returns whether the lattice changed.
  ChangeResult set(Attribute newEncoding, bool newMayVary) {
    if (isOverdefined)
      return ChangeResult::NoChange;

    // If we don't have an encoding yet, take the new one.
    if (!encoding) {
      encoding = newEncoding;
      mayVary = newMayVary;
      return ChangeResult::Change;
    }

    // Combine with existing encoding.
    return combine(newEncoding, newMayVary);
  }

  /// Mark as overdefined (conflicting encodings).
  ChangeResult markOverdefined() {
    if (isOverdefined)
      return ChangeResult::NoChange;
    isOverdefined = true;
    return ChangeResult::Change;
  }

  /// Join operation (used by forward analysis).
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    const auto &other = static_cast<const LayoutLattice &>(rhs);
    return combineWith(other);
  }

  /// Meet operation (used by backward analysis) - same as join for this
  /// analysis.
  ChangeResult meet(const AbstractSparseLattice &rhs) override {
    const auto &other = static_cast<const LayoutLattice &>(rhs);
    return combineWith(other);
  }

  Attribute encoding;
  bool mayVary = false;
  bool isOverdefined = false;

private:
  /// Combine with another lattice value.
  ChangeResult combineWith(const LayoutLattice &other) {
    // If other is uninitialized, no change.
    if (!other.encoding && !other.isOverdefined)
      return ChangeResult::NoChange;

    // If other is overdefined, we become overdefined.
    if (other.isOverdefined)
      return markOverdefined();

    // If we're uninitialized, take other's value.
    if (!encoding) {
      encoding = other.encoding;
      mayVary = other.mayVary;
      return ChangeResult::Change;
    }

    return combine(other.encoding, other.mayVary);
  }

  /// Combine this encoding with a new one.
  ChangeResult combine(Attribute newEncoding, bool newMayVary) {
    // If encodings match, prefer the harder constraint.
    if (encoding == newEncoding) {
      if (mayVary && !newMayVary) {
        mayVary = false;
        return ChangeResult::Change;
      }
      return ChangeResult::NoChange;
    }

    // Different encodings - soft constraints yield to others.
    if (mayVary) {
      encoding = newEncoding;
      mayVary = newMayVary;
      return ChangeResult::Change;
    }
    if (newMayVary) {
      // Keep our hard encoding.
      return ChangeResult::NoChange;
    }

    // Two different hard encodings - conflict!
    return markOverdefined();
  }
};

//===----------------------------------------------------------------------===//
// ForwardLayoutAnalysis
//===----------------------------------------------------------------------===//

/// Returns true if the operation produces encodings that may vary.
static bool encodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp, triton::CatOp,
             triton::TransOp>(op);
}

class ForwardLayoutAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LayoutLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const LayoutLattice *> operands,
                               ArrayRef<LayoutLattice *> results) override {
    // For operations without tensor results, nothing to do.
    if (results.empty())
      return success();

    // Find an operand with an encoding.
    const LayoutLattice *srcLattice = nullptr;
    for (const auto *operand : operands) {
      if (operand->hasEncoding()) {
        srcLattice = operand;
        break;
      }
    }

    if (!srcLattice)
      return success();

    // Infer the destination encoding from the source.
    Attribute dstEnc = inferDstEncoding(op, srcLattice->encoding);
    if (!dstEnc)
      return success();

    bool dstMayVary = srcLattice->mayVary || encodingsMayVary(op);

    // Propagate to all tensor results.
    for (auto [result, lattice] : llvm::zip(op->getResults(), results)) {
      if (!isa<RankedTensorType>(result.getType()))
        continue;
      propagateIfChanged(lattice, lattice->set(dstEnc, dstMayVary));
    }

    return success();
  }

  void setToEntryState(LayoutLattice *lattice) override {
    // Entry state is uninitialized - no change needed.
  }
};

//===----------------------------------------------------------------------===//
// BackwardLayoutAnalysis
//===----------------------------------------------------------------------===//

class BackwardLayoutAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<LayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<LayoutLattice *> operands,
                               ArrayRef<const LayoutLattice *> results) override {
    // For operations without tensor operands, nothing to do.
    if (operands.empty())
      return success();

    // Find a result with an encoding.
    const LayoutLattice *dstLattice = nullptr;
    for (const auto *result : results) {
      if (result->hasEncoding()) {
        dstLattice = result;
        break;
      }
    }

    if (!dstLattice)
      return success();

    // Infer the source encoding from the destination.
    Attribute srcEnc = inferSrcEncoding(op, dstLattice->encoding);
    if (!srcEnc)
      return success();

    bool srcMayVary = dstLattice->mayVary || encodingsMayVary(op);

    // Create a temporary lattice to meet with operands.
    LayoutLattice srcLattice(Value{});
    srcLattice.encoding = srcEnc;
    srcLattice.mayVary = srcMayVary;

    // Propagate to all tensor operands via meet.
    for (auto [operand, lattice] : llvm::zip(op->getOperands(), operands)) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      meet(lattice, srcLattice);
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    // Non-forwarded branch operands don't propagate layouts.
  }

  void visitCallOperand(OpOperand &operand) override {
    // Non-forwarded call operands don't propagate layouts.
  }

  void visitNonControlFlowArguments(RegionSuccessor &successor,
                                    ArrayRef<BlockArgument> arguments) override {
    // Non-control-flow arguments (e.g., loop induction variables) don't
    // propagate layouts.
  }

  void setToExitState(LayoutLattice *lattice) override {
    // Exit state is uninitialized - no change needed.
  }
};

} // namespace

LogicalResult inferLayout(
    FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
    const llvm::SmallVector<std::pair<Value, Attribute>> &seedEncodings) {
  // Disallow auto encoding across function call boundaries.
  for (auto argTy : func.getArgumentTypes()) {
    if (typeCheck(argTy)) {
      return func->emitError(
          "Functions taking auto encoding must be fully inlined");
    }
  }
  for (auto resultTy : func.getResultTypes()) {
    if (typeCheck(resultTy))
      return func->emitError(
          "Functions returning auto encoding must be fully inlined");
  }

  // Set up the dataflow solver.
  DataFlowSolver solver;
  SymbolTableCollection symbolTable;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<ForwardLayoutAnalysis>();
  solver.load<BackwardLayoutAnalysis>(symbolTable);

  // Initialize seed encodings before running the analysis.
  for (auto &[value, encoding] : seedEncodings) {
    auto *lattice = solver.getOrCreateState<LayoutLattice>(value);
    (void)lattice->set(encoding, /*mayVary=*/false);
  }

  // Run the analysis to fixed point.
  if (failed(solver.initializeAndRun(func)))
    return failure();

  // Check for conflicts and apply the inferred encodings.
  bool hadError = false;
  func.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (!typeCheck(result.getType()))
        continue;
      auto *lattice = solver.lookupState<LayoutLattice>(result);
      if (!lattice || !lattice->encoding) {
        // No encoding inferred - will be caught by doubleCheckEncodings.
        continue;
      }
      if (lattice->isOverdefined) {
        op->emitOpError("found conflicting encodings for result");
        hadError = true;
        continue;
      }
      // Apply the encoding.
      auto existingTy = cast<RankedTensorType>(result.getType());
      auto ty = existingTy.cloneWithEncoding(lattice->encoding);
      result.setType(ty);

      // Handle constant ops specially.
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        auto value = cast<SplatElementsAttr>(constantOp.getValueAttr());
        auto newValue =
            SplatElementsAttr::get(ty, value.getSplatValue<Attribute>());
        constantOp.setValueAttr(newValue);
      }
    }
  });

  // Also check block arguments.
  func.walk([&](Block *block) {
    for (auto arg : block->getArguments()) {
      if (!typeCheck(arg.getType()))
        continue;
      auto *lattice = solver.lookupState<LayoutLattice>(arg);
      if (!lattice || !lattice->encoding) {
        // No encoding inferred - will be caught by doubleCheckEncodings.
        continue;
      }
      if (lattice->isOverdefined) {
        block->getParentOp()->emitOpError(
            "found conflicting encodings for block argument");
        hadError = true;
        continue;
      }
      // Apply the encoding.
      auto existingTy = cast<RankedTensorType>(arg.getType());
      auto ty = existingTy.cloneWithEncoding(lattice->encoding);
      arg.setType(ty);
    }
  });

  if (hadError)
    return failure();

  return success();
}

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck) {
  auto res = mod.walk([&](Operation *op) -> WalkResult {
    for (auto resTy : op->getResultTypes()) {
      if (typeCheck(resTy)) {
        return op->emitOpError("Failed to infer return type");
      }
    }
    return success();
  });
  if (res.wasInterrupted())
    return failure();

  res = mod.walk([&](Block *block) -> WalkResult {
    for (auto argTy : block->getArgumentTypes()) {
      if (typeCheck(argTy)) {
        return block->getParentOp()->emitError(
            "Failed to infer block argument type");
      }
    }
    return success();
  });
  if (res.wasInterrupted())
    return failure();
  return success();
}

} // namespace mlir::triton::gluon
