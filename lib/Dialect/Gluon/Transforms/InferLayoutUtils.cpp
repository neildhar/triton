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
class LayoutLattice : public dataflow::AbstractSparseLattice {
public:
  Attribute encoding;
  enum class State {
    Undefined,
    MayVary,
    SingleEncoding,
    Overdefined,
  } state = State::Undefined;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutLattice)
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override {
    switch (state) {
    case State::Undefined:
      os << "Undefined";
    case State::MayVary:
      os << "MayVary";
    case State::SingleEncoding:
      os << "Single(" << encoding << ')';
    case State::Overdefined:
      os << "Overdefined";
    }
  }

  /// Join operation (used by forward analysis).
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    auto &other = static_cast<const LayoutLattice &>(rhs);
    return joinImpl(other.state, other.encoding);
  }

  /// Meet operation (used by backward analysis) - same as join for this
  /// analysis.
  ChangeResult meet(const AbstractSparseLattice &rhs) override {
    return join(rhs);
  }

  ChangeResult joinImpl(State otherState, Attribute otherEncoding) {
    // If the other lattice is in an earlier state, nothing to do.
    if (otherState < state)
      return ChangeResult::NoChange;

    // If both are fixed encodings, then we have failed.
    if (otherState == State::SingleEncoding && state == State::SingleEncoding)
      return setState(State::Overdefined, {});

    // Advance to the state of other.
    // Hash the string representation so conflict resolution does not depend on
    // solver visitation order.
    auto hashEncoding = [](Attribute attr) {
      std::string str;
      llvm::raw_string_ostream os(str);
      attr.print(os);
      return llvm::xxh3_64bits(str);
    };
    auto newEncoding = hashEncoding(encoding) > hashEncoding(otherEncoding)
                           ? encoding
                           : otherEncoding;
    return setState(otherState, newEncoding);
  }

private:
  ChangeResult setState(State newState, Attribute newEncoding) {
    auto oldState = std::exchange(state, newState);
    assert(newState >= oldState && "State going backwards");
    // TODO: If state does not change, but encoding does, is this sufficient? Is
    //       there a way to make it monotonic?
    auto oldEncoding = std::exchange(encoding, newEncoding);
    return oldState == newState && oldEncoding == newEncoding
               ? ChangeResult::NoChange
               : ChangeResult::Change;
  }
};

//===----------------------------------------------------------------------===//
// ForwardLayoutAnalysis
//===----------------------------------------------------------------------===//

/// Returns true if the operation produces encodings that may vary.
bool encodingsMayVary(Operation *op) {
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
    // Find an operand with an encoding.
    const LayoutLattice *srcLattice = nullptr;
    for (const auto *operand : operands) {
      if (operand->encoding) {
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

    bool dstMayVary = encodingsMayVary(op);
    auto dstState =
        dstMayVary ? LayoutLattice::State::MayVary : srcLattice->state;
    // Create a temporary lattice to meet with operands.
    LayoutLattice dstLattice(Value{});
    dstLattice.encoding = dstEnc;
    dstLattice.state = dstState;

    // Propagate to all tensor results.
    for (auto [result, lattice] : llvm::zip(op->getResults(), results)) {
      if (!isa<RankedTensorType>(result.getType()))
        continue;

      propagateIfChanged(lattice, lattice->join(dstLattice));
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

  LogicalResult
  visitOperation(Operation *op, ArrayRef<LayoutLattice *> operands,
                 ArrayRef<const LayoutLattice *> results) override {
    // Find a result with an encoding.
    const LayoutLattice *dstLattice = nullptr;
    for (const auto *result : results) {
      if (result->encoding) {
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

    bool srcMayVary = encodingsMayVary(op);
    auto srcState =
        srcMayVary ? LayoutLattice::State::MayVary : dstLattice->state;

    // Create a temporary lattice to meet with operands.
    LayoutLattice srcLattice(Value{});
    srcLattice.encoding = srcEnc;
    srcLattice.state = srcState;

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

  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
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
    (void)lattice->joinImpl(LayoutLattice::State::SingleEncoding, encoding);
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
