#define DEBUG_TYPE "eddi"

#include <set>
#include <string>
#include <map>
#include <vector>

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Instrumentation.h"

using namespace llvm;

STATISTIC(EDDIBranchesAdded,
          "Number of EDDI sections");

static cl::opt<unsigned> InstMax("eddi-inst-max",
                            cl::desc("Maximum number of instructions to allow without EDDI check"),
                            cl::value_desc("value"),
                            cl::init(10));
static cl::opt<unsigned> InstMin("eddi-inst-min",
                            cl::desc("Mininmum number of instructions before adding an EDDI check"),
                            cl::value_desc("value"),
                            cl::init(1));
static cl::opt<bool> DupLoads("eddi-dup-loads",
                            cl::desc("Whether or not EDDI should duplicate loads"),
                            cl::value_desc("true/false"),
                            cl::init(true));
static cl::opt<bool> CheckPtrs("eddi-check-ptrs",
                            cl::desc("Whether or not EDDI should check pointer values"),
                            cl::value_desc("true/false"),
                            cl::init(true));



namespace {
  class EDDI : public ModulePass {
    public:
      static char ID;
      EDDI() : ModulePass(ID) {
        initializeEDDIPass(*PassRegistry::getPassRegistry());
      }

      virtual bool runOnModule(Module &M);
      virtual void getAnalysisUsage(AnalysisUsage &usage) const;

    private:
      const static std::string failBlockName;

      BasicBlock *addFailBlock(Module &M, Function &F, GlobalVariable *logFormat);
      void duplicateBlocks(std::vector<BasicBlock *> &blocks, BasicBlock *failBlock, Function &F);
      BasicBlock *duplicateBlock(BasicBlock *currentBlock, BasicBlock *failBlock, Function &F);
      void addComparisonBlocks(BasicBlock *origBlock, BasicBlock *nextBlock,
                               BasicBlock *dupBlock, BasicBlock *failBlock);
      bool canDuplicate(Instruction *inst);
      bool canDuplicateRange(BasicBlock::iterator BI, BasicBlock::iterator BE);
      Instruction *findComparisonInst(BasicBlock *block);
  };

  const std::string EDDI::failBlockName = "eddiFail";
}

bool EDDI::runOnModule(Module &M) {
  // Create a global variable to store the printf format for logging errors
  std::string logString = "EDDI: Data inconsistency in %s\n";
  Constant *logFormat = ConstantDataArray::getString(M.getContext(), logString);
  GlobalVariable *logFormatVar = new GlobalVariable(M, logFormat->getType(),
      true, GlobalValue::LinkOnceAnyLinkage, logFormat, "");

  std::vector<BasicBlock *> blocks;

  Module::iterator MI = M.begin();
  Module::iterator ME = M.end();
  for (; MI != ME; MI++) {
    Function &F = *MI;
    // Do not operate on function declarations
    if (F.isDeclaration())
      continue;

    LoopInfo &LI = getAnalysis<LoopInfo>(F);

    // Iterate across all basic blocks in the function
    Function::iterator FI = F.begin();
    Function::iterator FE = F.end();
    for (; FI != FE; FI++) {
      blocks.push_back(FI);
    }

    // Add fail block to this function
    BasicBlock *failBlock = addFailBlock(M, F, logFormatVar);

    duplicateBlocks(blocks, failBlock, F);
    blocks.clear();
  }
  return true;
}

void EDDI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<DominatorTree>();
}

/// addFailBlock - Adds a new BasicBlock to the end of the given function
/// that will be branched to upon detection of an EDDI failure
///
BasicBlock *EDDI::addFailBlock(Module &M, Function &F, GlobalVariable *logFormat) {
  BasicBlock *failBlock = BasicBlock::Create(F.getContext(), failBlockName, &F);

  IRBuilder<> builder(failBlock);

  /* Acquire reference to stdlib printf function */
  Type *printfParams[1];
  printfParams[0] = Type::getInt8PtrTy(M.getContext());
  ArrayRef<Type *> paramsRef(printfParams);
  FunctionType *printfType = FunctionType::get(Type::getInt32Ty(M.getContext()), paramsRef, true);
  Constant *printfFunc = M.getOrInsertFunction("printf", printfType);

  /* Acquire reference to stdlib exit function */
  Type *exitParams[1];
  exitParams[0] = Type::getInt32Ty(M.getContext());
  ArrayRef<Type *> exitParamsRef(exitParams);
  FunctionType *exitType = FunctionType::get(Type::getVoidTy(M.getContext()), exitParamsRef, false);
  Constant *exitFunc = M.getOrInsertFunction("exit", exitType);

  /* Used for getelementptrinst, which is needed to pass global strings into printf */
  Value *elementPtrIndices[2];
  elementPtrIndices[0] = elementPtrIndices[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false);
  ArrayRef<Value *> ptrIndicesRef(elementPtrIndices);

  /* Put the current function's name as a global variable */
  Constant *funcName = ConstantDataArray::getString(M.getContext(), F.getName());
  GlobalVariable *funcNameVar = new GlobalVariable(M, funcName->getType(),
      true, GlobalVariable::InternalLinkage,
      funcName, "");

  Value *formatGetInst = builder.CreateGEP(logFormat, ptrIndicesRef);
  Value *funcNameGetInst = builder.CreateGEP(funcNameVar, ptrIndicesRef);
  builder.CreateCall2(printfFunc, formatGetInst, funcNameGetInst);
  builder.CreateCall(exitFunc, ConstantInt::get(Type::getInt32Ty(F.getContext()), 1, true));
  builder.CreateUnreachable();

  return failBlock;
}

/// duplicateBlocks - duplicates all supplied BasicBlocks. CallInst and
/// InvokeInst instructions will not be duplicated. A reference to a StoreInst
/// will trigger the immediate addition of EDDI checks.
///
void EDDI::duplicateBlocks(std::vector<BasicBlock *> &blocks, BasicBlock *failBlock, Function &F) {
  std::vector<BasicBlock *>::iterator VI = blocks.begin();
  std::vector<BasicBlock *>::iterator VE = blocks.end();

  for (; VI != VE; VI++) {
    BasicBlock *currentBlock = *VI;

    BasicBlock *tail = duplicateBlock(currentBlock, failBlock, F);
    while (tail != NULL) {
      // A tail means that this BasicBlock was split and still has instructions
      // that need EDDI checking applied
      tail = duplicateBlock(tail, failBlock, F);
    }
  }
}

BasicBlock *EDDI::duplicateBlock(BasicBlock *currentBlock, BasicBlock *failBlock, Function &F) {
  BasicBlock::iterator BI_start = currentBlock->begin();
  BasicBlock::iterator BI_end = currentBlock->begin();
  BasicBlock::iterator BE = currentBlock->end();

  unsigned int instructionCount = 0; // the number of instructions so far before an EDDI
                                     // check has occurred
  BasicBlock *tail = NULL;

  for (; BI_end != BE; BI_end++) {
    Instruction *currentInst = BI_end;
    instructionCount++;

    StoreInst *storeInst = dyn_cast<StoreInst>(currentInst);
    CallInst *callInst = dyn_cast<CallInst>(currentInst);

    if ((storeInst != NULL || callInst != NULL) &&
        BI_start == BI_end) {
      // Since LLVM doesn't treat these as terminator instructions,
      // if we find one of these at the beginning of a BasicBlock
      // then we can just continue processing the block as normal,
      // as it means that all instructions prior to this have been
      // checked
      continue;
    }

    bool isTerminator = (currentInst == currentBlock->getTerminator());

    // Hit the instruction limit, or at the end of a StoreBasicBlock
    // Duplicate the block up to this point if there is at least
    // one instruction that we can duplicate
    if ((instructionCount == InstMax ||
        isTerminator ||
        storeInst != NULL ||
        callInst != NULL) && canDuplicateRange(BI_start, BI_end)) {
      instructionCount = 0;

      BasicBlock *splitBlock = currentBlock->splitBasicBlock(BI_end);

      if (!isTerminator)
        tail = splitBlock;

      // Splitting the block broke our iterators, so fix those before
      // continuing
      BasicBlock::iterator BI_dup = currentBlock->begin();
      BE = currentBlock->end();
      BE--; // Move back to the last instruction (don't duplicate terminator)

      // Create the BasicBlock where we will insert duplicated instructions
      BasicBlock *checkBlock = BasicBlock::Create(F.getContext(), "", &F, splitBlock);

      // Patch the current block's unconditional branch to point to the
      // checkBlock instead
      BranchInst *origBranch = dyn_cast<BranchInst>(currentBlock->getTerminator());
      if (origBranch == NULL) {
        errs() << "Failed to patch branch for split block\n";
        break;
      }
      else {
        origBranch->setSuccessor(0, checkBlock);
      }

      // The instruction map is used for remapping instruction operands.
      // This way, we can ensure that the cloned instructions use cloned
      // operands if available
      //
      // The usage map is needed to determine which instruction values have
      // been remapped, so we can compare the original and duplicated values.
      std::map<Instruction *, Instruction *> instMap;

      for (; BI_dup != BE; BI_dup++) {
        Instruction *dupInst = BI_dup;
        if (!canDuplicate(dupInst))
          continue;

        Instruction *newInst = dupInst->clone();
        instMap.insert(std::pair<Instruction *, Instruction *>(dupInst, newInst));
        checkBlock->getInstList().push_back(newInst);

        // See if this instruction references any earlier ones in the instMap
        unsigned int numOperands = newInst->getNumOperands();
        for (unsigned int i = 0; i < numOperands; i++) {
          Value *operand = newInst->getOperand(i);
          
          // Only check the map if the operand is an instruction
          if (Instruction *origInst = dyn_cast<Instruction>(operand)) {
            std::map<Instruction *, Instruction *>::iterator findResult;
            findResult = instMap.find(origInst);

            if (findResult != instMap.end()) {
              // Found the instruction - substitute it in
              Instruction *dupOperand = (*findResult).second;
              newInst->setOperand(i, dupOperand);
            }
          }
        }
      }

      // checkBlock now contains the duplicated block - now to generate
      // comparison instructions
      addComparisonBlocks(currentBlock, splitBlock, checkBlock, failBlock);
      break;
    }
  }

  return tail;
}

/// addComparisonBlocks - adds the BasicBlocks for comparing the values of
/// origBlock and dupBlock
///
void EDDI::addComparisonBlocks(BasicBlock *origBlock, BasicBlock *nextBlock,
                               BasicBlock *dupBlock, BasicBlock *failBlock) {
  Function *F = origBlock->getParent();

  std::vector<BasicBlock *> comparisonBlocks;

  BasicBlock::iterator origII = origBlock->begin();
  BasicBlock::iterator origIE = origBlock->end();
  BasicBlock::iterator dupII = dupBlock->begin();
  BasicBlock::iterator dupIE = dupBlock->end();

  for (; origII != origIE; origII++) {
    // There may be instructions that we did not duplicate remaining in the
    // original block. Skip these instructions.
    if (!canDuplicate(origII))
      continue;

    Instruction *origInst = origII;
    Instruction *dupInst = dupII;
    Type *instType = origInst->getType();
    BasicBlock *compareBlock = BasicBlock::Create(F->getContext(), "", F);
    comparisonBlocks.push_back(compareBlock);
    IRBuilder<> builder(compareBlock);

    if (instType->isFloatingPointTy()) {
      if (dupInst->getType() != instType) {
        errs() << "Instruction types do not match\n";
        return;
      }
      builder.CreateFCmpUNE(origInst, dupInst);
    }
    else if (instType->isIntegerTy()) {
      if (dupInst->getType() != instType) {
        errs() << "Instruction types do not match\n";
        return;
      }
      builder.CreateICmpNE(origInst, dupInst);
    }
    else if (instType->isPointerTy()) {
      if (dupInst->getType() != instType) {
        errs() << "Instruction types do not match\n";
        return;
      }
      // Pointer types require integer conversions first
      // If we pick a 64-bit integer, we're fine on any architecture. See
      // http://llvm.org/docs/GetElementPtr.html#how-is-gep-different-from-ptrtoint-arithmetic-and-inttoptr
      Type *int64Type = Type::getInt64Ty(F->getContext());
      Value *origPtrInt = builder.CreatePtrToInt(origInst, int64Type);
      Value *dupPtrInt = builder.CreatePtrToInt(dupInst, int64Type);
      builder.CreateICmpNE(origPtrInt, dupPtrInt);
    }
    dupII++;
  }

  // Hook the dupBlock into the first comparison block
  Function::BasicBlockListType &blockList = F->getBasicBlockList();
  Function::BasicBlockListType::iterator BBI = blockList.begin();
  Function::BasicBlockListType::iterator BBE = blockList.end();

  for (; BBI != BBE; BBI++) {
    if (&(*BBI) == nextBlock)
      break;
  }
  if (BBI == BBE) {
    errs() << "Did not find nextBlock\n";
    return;
  }

  unsigned int numBlocks = comparisonBlocks.size();
  IRBuilder<> dupTermBuilder(dupBlock);

  if (numBlocks > 0) {
    dupTermBuilder.CreateBr(comparisonBlocks[0]);
  }
  else {
    dupTermBuilder.CreateBr(nextBlock);
  }

  for (unsigned int i = 0; i < numBlocks; i++) {
    // Insert each comparison block into the function, and add branches
    BasicBlock *curComparisonBlock = comparisonBlocks[i];
    Instruction *comparison = findComparisonInst(curComparisonBlock);

    curComparisonBlock->moveBefore(BBI);
    IRBuilder<> curBlockBuilder(curComparisonBlock);

    if (i == numBlocks - 1) {
      curBlockBuilder.CreateCondBr(comparison, failBlock, nextBlock);
    }
    else {
      curBlockBuilder.CreateCondBr(comparison, failBlock, comparisonBlocks[i+1]);
    }

    EDDIBranchesAdded++;
  }
}

bool EDDI::canDuplicate(Instruction *inst) {
  // BasicBlock terminators should not be duplicated, and function calls count
  // as terminators whether LLVM wants them to or not.
  //
  // Other instructions that should not be duplicated, at least without further
  // thought: PHINode, AtomicRMWInst, AtomicCmpXchgInst, ShuffleVectorInst, 
  // VAArgInst
  if (isa<CallInst>(inst) ||
      isa<TerminatorInst>(inst) ||
      isa<PHINode>(inst) ||
      isa<AtomicRMWInst>(inst) ||
      isa<AtomicCmpXchgInst>(inst) ||
      isa<ShuffleVectorInst>(inst) ||
      isa<VAArgInst>(inst))
    return false;

  if (!DupLoads && isa<LoadInst>(inst))
    return false;

  Type *instType = inst->getType();
  if (instType->isFloatingPointTy())
    return true;
  if (instType->isIntegerTy())
    return true;
  if (instType->isPointerTy() &&
      CheckPtrs &&
      !isa<AllocaInst>(inst))
    return true;

  return false;
}

bool EDDI::canDuplicateRange(BasicBlock::iterator BI, BasicBlock::iterator BE) {
  BasicBlock::iterator blockIter;
  for (blockIter = BI; blockIter != BE; blockIter++) {
    if (canDuplicate(BI))
      return true;
  }

  return false;
}

Instruction *EDDI::findComparisonInst(BasicBlock *block) {
  BasicBlock::iterator BI = block->begin();
  BasicBlock::iterator BE = block->end();

  for (; BI != BE; BI++) {
    Instruction *inst = BI;
    if (isa<ICmpInst>(inst) || isa<FCmpInst>(inst))
      return inst;
  }

  return NULL;
}

char EDDI::ID = 0;
INITIALIZE_PASS_BEGIN(EDDI, 
                "eddi",
                "Error Detection by Duplicated Instructions",
                false,
                false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(EDDI, 
                "eddi",
                "Error Detection by Duplicated Instructions",
                false,
                false)
