#define DEBUG_TYPE "cfcss"

#include <set>
#include <string>
#include <map>
#include <vector>

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Type.h"

using namespace llvm;

STATISTIC(NumBlocksFlowChecked,
          "Number of basic blocks with control-flow checking added");

namespace {
  /* A node in the control-flow graph. This class contains auxillary
   * information that is needed in order to complete the CFCSS
   * transformation.
   */
  class CFGraphNode {
    private:
      BasicBlock *block;
      uint64_t signature;
      bool fanInNode;
      std::vector<CFGraphNode *> predecessors;
      std::vector<CFGraphNode *> successors;
    public:
      CFGraphNode(BasicBlock *block_) : block(block_) { this->fanInNode = false; }
      void addPred(CFGraphNode *pred) { predecessors.push_back(pred); }
      void addSucc(CFGraphNode *succ) { successors.push_back(succ); }
      BasicBlock *getBasicBlock() { return block; }
      void setSignature(long signature_) { this->signature = signature_; }
      long getSignature() { return this->signature; }
      void setFanInNode(bool fanInNode_) { this->fanInNode = fanInNode_; }
      long isFanInNode() { return this->fanInNode; }
      bool callsParentFunctionOfNode(CFGraphNode *node);

      std::vector<CFGraphNode *> &getPreds() { return predecessors; }
      std::vector<CFGraphNode *> &getSuccs() { return successors; }
  };

  bool CFGraphNode::callsParentFunctionOfNode(CFGraphNode *node) {
    Function *parentFunc = node->getBasicBlock()->getParent();
    BasicBlock::iterator II = block->begin();
    BasicBlock::iterator IE = block->end();

    for (; II != IE; II++) {
      if (CallInst *inst = dyn_cast<CallInst>(II)) {
        if (inst->getCalledFunction() == parentFunc) {
          return true;
        }
      }
    }
    return false;
  }

  class CFGraphEdge {
    private:
      BasicBlock *from;
      BasicBlock *to;
    public:
      CFGraphEdge(BasicBlock *from_, BasicBlock *to_) : from(from_), to(to_) {}
      friend bool operator<(const CFGraphEdge &e1, const CFGraphEdge &e2);
      BasicBlock *getFrom() const { return from; }
      BasicBlock *getTo() const { return to; }
  };

  bool operator<(const CFGraphEdge &e1, const CFGraphEdge &e2) {
    if (e1.from < e2.from)
      return true;
    else
      return e1.to < e2.to;
  }

  class CFCSS : public ModulePass {
    public:
      static char ID;
      CFCSS() : ModulePass(ID) {
        initializeCFCSSPass(*PassRegistry::getPassRegistry());
      }

    private:
      /* Name given to the block added to each function, which is called
       * in event of a CFCSS failure */
      const static std::string failBlockName;
      /* Set of BasicBlock *, which form the vertices of the CFG */
      std::set<BasicBlock *> *vertices;

      /* Set of CFGraphEdge, which form the edges of the CFG */
      std::set<CFGraphEdge> *edges;

      /* Map from Function * to a vector of BasicBlocks;
       * this allows us to find the edges from return instructions
       * back to block after the caller */
      std::map<Function *, std::vector<BasicBlock *> > *callers;

      /* Map of BasicBlocks to CFGraphNodes, which contain
       * auxillary information about vertices */
      std::map<BasicBlock *, CFGraphNode *> *bbMap;

      /* The CFGraphNodes in the CFG; these contain additional information
       * about the nodes, including their CFCSS signatures */
      std::vector<CFGraphNode *> *verticesData;

      /* The BasicBlocks that are not to be flow checked */
      std::set<BasicBlock *> *noCheckBlocks;

      /* Inserts the CFCSS error-handling block into each function
       * in the module */
      void insertFailBlocks(Module &M);
      
      /* Splits blocks containing function calls, as CallInst is not
       * considered a basic block terminator. To get around this,
       * we split a given block at a CallInst, then pretend that
       * the return from that function goes to the following block */
      void splitBlocks();

      /* The set of basic blocks in the program form the vertices of the
       * CFG; this method creates the edges */
      void createEdges();
      
      /* Take the set of basic blocks and assign each to a CFGraphNode,
       * which will let us assign signatures and determine if the basic
       * block is a fan-in node or not */
      void buildGraph();

      /* Generate the CFCSS verification instructions */
      void generateInstructions(Module &M);

      /* Finds the CFCSS fail block for a given function */
      BasicBlock *findFailBlock(Function *f);

      /* Gets the set of vertices this block has outgoing
       * connections to */
      std::set<BasicBlock *> getToVertices(BasicBlock *block);

      /* Gets the set of vertices this block has incoming
       * connections from */
      std::set<BasicBlock *> getFromVertices(BasicBlock *block);
 
      virtual bool runOnModule(Module &M) {
        this->vertices = new std::set<BasicBlock *>();
        this->edges = new std::set<CFGraphEdge>();
        this->callers = new std::map<Function *, std::vector<BasicBlock *> >();
        this->bbMap = new std::map<BasicBlock *, CFGraphNode *>();
        this->verticesData = new std::vector<CFGraphNode *>();
        this->noCheckBlocks = new std::set<BasicBlock *>();

        /* Perform the transformation */
        this->insertFailBlocks(M);
        this->splitBlocks();
        this->createEdges();
        this->buildGraph();
        this->generateInstructions(M);

        delete vertices;
        delete edges;
        delete callers;
        delete bbMap;
        delete noCheckBlocks;

        std::vector<CFGraphNode *>::iterator NI = verticesData->begin();
        std::vector<CFGraphNode *>::iterator NE = verticesData->end();

        for (; NI != NE; NI++) {
          delete *NI;
        }

        delete verticesData;

        return true;
      }
  };

  /* Inserts the CFCSS error-handling block into each function
   * in the module */
  void CFCSS::insertFailBlocks(Module &M) {
    /* Format string for logging a detected control-flow error */
    std::string logString = "Control flow error detected in %s\n";
    Constant *logFormat = ConstantDataArray::getString(M.getContext(), logString);
    GlobalVariable *logFormatVar = new GlobalVariable(M, logFormat->getType(),
        true, GlobalValue::InternalLinkage,
        logFormat, "");

    /* Acquire reference to stdlib printf function */
    Type *printfParams[1];
    printfParams[0] = Type::getInt8PtrTy(M.getContext());
    ArrayRef<Type *> paramsRef(printfParams);
    FunctionType *printfType = FunctionType::get(Type::getInt32Ty(M.getContext()), paramsRef, true);
    Constant *printf = M.getOrInsertFunction("printf", printfType);

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


    Module::iterator MI = M.begin();
    Module::iterator ME = M.end();

    for (; MI != ME; MI++) {
      /* For every function, add a new basic block at the end
       * that is branched to if we encounter a control flow error */

      /* Function is not linked in yet; do not add control flow checking */
      if (MI->size() == 0)
        continue;

      /* Put the current function's name as a global variable */
      Constant *funcName = ConstantDataArray::getString(M.getContext(), MI->getName());
      GlobalVariable *funcNameVar = new GlobalVariable(M, funcName->getType(),
          true, GlobalVariable::InternalLinkage,
          funcName, "");

      /* Insert the fail block at the end of the current function */
      BasicBlock *failBlock = BasicBlock::Create(M.getContext(), failBlockName, MI);

      /* Instructions to get the strings for printf */
      Value *formatGetInst = GetElementPtrInst::Create(logFormatVar, ptrIndicesRef, "", failBlock);
      Value *funcNameGetInst = GetElementPtrInst::Create(funcNameVar, ptrIndicesRef, "", failBlock);

      /* Put together the arguments for printf */
      std::vector<Value *> printfArgs;
      printfArgs.push_back(formatGetInst);
      printfArgs.push_back(funcNameGetInst);

      /* Add the printf instruction */
      CallInst::Create(printf, printfArgs, "", failBlock);

      /* Set up arguments to the exit call */
      std::vector<Value *> exitArgs;
      exitArgs.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()), 1, true));

      CallInst::Create(exitFunc, exitArgs, "", failBlock);
      
      /* The program terminates when exit is called, so this is our terminator
       * for the basic block */
      new UnreachableInst(M.getContext(), failBlock);

      Function::iterator FI = MI->begin();
      Function::iterator FE = MI->end();

      /* Add basic blocks in this function to the vertices set */
      for (; FI != FE; FI++) {
        if (FI->getName() != failBlockName)
          this->vertices->insert(FI);
      }
    }
  }

  /* Splits blocks containing function calls, as CallInst is not
   * considered a basic block terminator. To get around this,
   * we split a given block at a CallInst, then pretend that
   * the return from that function goes to the following block */
  void CFCSS::splitBlocks() {
    /* Basic blocks in toProcess potentially have a CallInst within
     * that needs to be split */
    std::vector<BasicBlock *> toProcess;
    std::set<BasicBlock *>::iterator VI = vertices->begin();
    std::set<BasicBlock *>::iterator VE = vertices->end();

    for (; VI != VE; VI++) {
      toProcess.push_back(*VI);
    }

    for (size_t i = 0; i < toProcess.size(); i++) {
      BasicBlock *curBlock = toProcess[i];

      BasicBlock::iterator II = curBlock->begin();
      BasicBlock::iterator IE = curBlock->end();

      for (; II != IE; II++) {
        if (CallInst *inst = dyn_cast<CallInst>(II)) {
          Function *calledFunc = inst->getCalledFunction();

          /* Do not bother splitting the block if we're calling an unlinked
           * function - we can't check the control flow for those functions
           * anyway */
          if (calledFunc->empty())
            continue;

          /* This block should be split after this instruction */
          BasicBlock *split = curBlock->splitBasicBlock(++II);
          toProcess.push_back(split);
          this->vertices->insert(split);
          this->noCheckBlocks->insert(curBlock);

          /* Insert an edge from the calling block to the called block */
          CFGraphEdge newEdge(curBlock, &calledFunc->getEntryBlock());
          edges->insert(newEdge);

          /* Store the next block after this function is called, as
           * we will later need to make an edge from any ReturnInst's
           * in the function to the subsequent block */
          std::map<Function *, std::vector<BasicBlock *> >::iterator entry;
          entry = this->callers->find(calledFunc);
          if (entry == this->callers->end()) {
            /* Insert a new entry vector */
            std::vector<BasicBlock *> entries;
            entries.push_back(split);
            (*callers)[calledFunc] = entries;
          }
          else {
            /* Append to current entries vector */
            entry->second.push_back(split);
          }
          break;
        }
      }
    }
  }

  /* The set of basic blocks in the program form the vertices of the
   * CFG; this method creates the edges */
  void CFCSS::createEdges() {
    std::set<BasicBlock *>::iterator VI = vertices->begin();
    std::set<BasicBlock *>::iterator VE = vertices->end();

    for (VI = vertices->begin(); VI != VE; VI++) {
      if ((*VI)->getName() == failBlockName)
        continue;

      TerminatorInst *terminator = (*VI)->getTerminator();
      if (terminator == NULL)
        continue;

      /* Based on the type of the terminator, we can find all possible
       * destinations */
      if (BranchInst *inst = dyn_cast<BranchInst>(terminator)) {
        int numSuccessors = inst->getNumSuccessors();
        
        /* CallInst followed by an unconditional BranchInst means a previously
         * split node.
         * getPrevNode will segfault if called on a block that has only one
         * instruction, so check to make sure that is not the case */
//        if (numSuccessors == 1 && &(*VI)->front() != inst &&
//            dyn_cast_or_null<CallInst>(inst->getPrevNode())) {
        if (noCheckBlocks->find(*VI) != noCheckBlocks->end()) {
          continue;
        }
        for (int i = 0 ; i < numSuccessors; i++) {
          BasicBlock *block = inst->getSuccessor(i);
          CFGraphEdge newEdge(*VI, block);
          edges->insert(newEdge);
        }
      }
      else if (IndirectBrInst *inst = dyn_cast<IndirectBrInst>(terminator)) {
        int numDests = inst->getNumDestinations();
        for (int i = 0 ; i < numDests; i++) {
          BasicBlock *block = inst->getDestination(i);
          CFGraphEdge newEdge(*VI, block);
          edges->insert(newEdge);
        }
      }
      else if (InvokeInst *inst = dyn_cast<InvokeInst>(terminator)) {
        Function *invokedFunc = inst->getCalledFunction();
        if (invokedFunc->empty())
          continue;

        BasicBlock &block = invokedFunc->getEntryBlock();
        CFGraphEdge newEdge(*VI, &block);
        edges->insert(newEdge);
      }
      else if (ReturnInst *inst = dyn_cast<ReturnInst>(terminator)) {
        BasicBlock *parentBlock = inst->getParent();
        if (parentBlock == NULL)
          continue;

        Function *parentFunc = parentBlock->getParent();
        if (parentFunc == NULL)
          continue;

        std::map<Function *, std::vector<BasicBlock *> >::iterator entry;
        entry = this->callers->find(parentFunc);
        if (entry != this->callers->end()) {
          std::vector<BasicBlock *> &returnBlocks = entry->second;

          for (size_t i = 0; i < returnBlocks.size(); i++) { 
            CFGraphEdge newEdge(*VI, returnBlocks[i]);
            edges->insert(newEdge);
          }
        }
      }
      else if (SwitchInst *inst = dyn_cast<SwitchInst>(terminator)) {
        int numSuccessors = inst->getNumSuccessors();
        for (int i = 0 ; i < numSuccessors; i++) {
          BasicBlock *block = inst->getSuccessor(i);
          CFGraphEdge newEdge(*VI, block);
          edges->insert(newEdge);
        }
      }

    }
  }

  /* Take the set of basic blocks and assign each to a CFGraphNode,
   * which will let us assign signatures and determine if the basic
   * block is a fan-in node or not */
  void CFCSS::buildGraph() {
    std::set<BasicBlock *>::iterator vertexIter = this->vertices->begin();
    std::set<BasicBlock *>::iterator vertexEnd = this->vertices->end();

    /* Map BasicBlocks to graph nodes */
    for (; vertexIter != vertexEnd; vertexIter++) {
      BasicBlock *block = *vertexIter;
      CFGraphNode *node = new CFGraphNode(block);
      (*bbMap)[block] = node;
      this->verticesData->push_back(node);
    }

    std::vector<CFGraphNode *>::iterator nodeIter = this->verticesData->begin();
    std::vector<CFGraphNode *>::iterator nodeEnd = this->verticesData->end();

    long signature = 1;

    /* Assign signatures to vertices, and set up successors/predecessors to
     * nodes in the CFG */
    for (; nodeIter != nodeEnd; nodeIter++) {
      CFGraphNode *node = *nodeIter;
      BasicBlock *block = node->getBasicBlock();
      std::set<BasicBlock *> toVertices = getToVertices(block);
      std::set<BasicBlock *>::iterator connectedBlockIter = toVertices.begin();
      std::set<BasicBlock *>::iterator connectedBlockEnd = toVertices.end();

      for (; connectedBlockIter != connectedBlockEnd; connectedBlockIter++) {
        node->addSucc((*bbMap)[*connectedBlockIter]);
      }

      std::set<BasicBlock *> fromVertices = getFromVertices(block);
      connectedBlockIter = fromVertices.begin();
      connectedBlockEnd = fromVertices.end();

      for (; connectedBlockIter != connectedBlockEnd; connectedBlockIter++) {
        node->addPred((*bbMap)[*connectedBlockIter]);
      }
      node->setSignature(signature);
      signature++;
    }

    /* Mark any node with more than one predecessor as a fan-in node */
    nodeIter = this->verticesData->begin();
    for (; nodeIter != nodeEnd; nodeIter++) {
      CFGraphNode *node = *nodeIter;
      std::vector<CFGraphNode *> &preds = node->getPreds();
      if (preds.size() > 1) {
        node->setFanInNode(true);
      }
    }
  }

  /* Generate the CFCSS verification instructions */
  void CFCSS::generateInstructions(Module &M) {
    LLVMContext &context = M.getContext();
    TargetData td(&M);

    /* Create global integers for D (runtime adjusting signature) and G
     * (signature of the current CFG) */
    IntegerType *intType = td.getIntPtrType(context);
    ConstantInt *initialD = ConstantInt::get(intType, 0);
    GlobalVariable *D = new GlobalVariable(M, intType, false, GlobalValue::InternalLinkage, initialD, "__CFCSS_D", NULL, true);
    ConstantInt *initialG = ConstantInt::get(intType, 0);
    GlobalVariable *G = new GlobalVariable(M, intType, false, GlobalValue::InternalLinkage, initialG, "__CFCSS_G", NULL, true);

    std::vector<CFGraphNode *>::iterator nodeIter = this->verticesData->begin();
    std::vector<CFGraphNode *>::iterator nodeEnd = this->verticesData->end();

    for (; nodeIter != nodeEnd; nodeIter++) {
      CFGraphNode *node = *nodeIter;
      std::vector<CFGraphNode *> &preds = node->getPreds();

      if (preds.size() > 0) {
        NumBlocksFlowChecked++;

        BasicBlock *block = node->getBasicBlock();
        BasicBlock *failBlock = findFailBlock(block->getParent());
        if (failBlock == NULL) {
          errs() << "Could not find fail block for node; continuing\n";
          continue;
        }
        /* Insert instructions for loading, calculating, and storing G */
        Instruction *firstInst = block->getFirstNonPHIOrDbg();
        CFGraphNode *pred = preds[0];
        uint64_t diffSig = pred->getSignature() ^ node->getSignature();
        ConstantInt *diffSigConst = ConstantInt::get(intType, diffSig);
        LoadInst *loadG = new LoadInst(G, "loadG", firstInst);
        BinaryOperator *xorOp = BinaryOperator::Create(Instruction::Xor, diffSigConst, loadG, Twine(), firstInst);
        BinaryOperator *adjOp = NULL;

        /* Use D as our runtime adjusting signature */
        if (node->isFanInNode()) {
          LoadInst *loadD = new LoadInst(D, "loadD", firstInst);
          adjOp = BinaryOperator::Create(Instruction::Xor, xorOp, loadD, Twine(), firstInst);
        }
        new StoreInst((adjOp ? adjOp : xorOp), G, firstInst);

        uint64_t expectedSig = node->getSignature();
        ConstantInt *expectedSigConst = ConstantInt::get(intType, expectedSig);

        ICmpInst *compareSigInst = new ICmpInst(firstInst, CmpInst::ICMP_EQ, (adjOp ? adjOp : xorOp), expectedSigConst);

        if (node->isFanInNode()) {
          /* Add additional instructions for runtime adjusting signature */
          for (size_t i = 0; i < preds.size(); i++) {
            /* Insert D calculation in predecessors */
            CFGraphNode *curPred = preds[i];
            uint64_t adjustSig = pred->getSignature() ^ curPred->getSignature();
            ConstantInt *adjustSigConst = ConstantInt::get(intType, adjustSig);
            
            BasicBlock *curPredBlock = curPred->getBasicBlock();
            if (curPred->callsParentFunctionOfNode(node)) {
              /* Insert this StoreInst before the call to the current node's
               * function */
              new StoreInst(adjustSigConst, D, curPredBlock->getFirstNonPHIOrDbg());
            }
            else {
              new StoreInst(adjustSigConst, D, curPredBlock->getTerminator());
            }
          }
        }
        /* Find the iterator to split at */
        BasicBlock::iterator BI = block->begin();
        BasicBlock::iterator BE = block->end();

        for (; BI != BE; BI++) {
          if (&(*BI) == firstInst)
            break;
        }
        if (BI == BE) {
          errs() << "Could not find instruction in basic block\n";
          continue;
        }

        BasicBlock *splitBlock = block->splitBasicBlock(BI);
        block->getTerminator()->eraseFromParent();
        BranchInst::Create(splitBlock, failBlock, compareSigInst, block);
      }
      else {
        /*
         * No predecessors. This is likely a special function like main, so
         * just assign G to this node's signature and don't perform any
         * checking.
         * */
        uint64_t nodeSig = node->getSignature();
        ConstantInt *nodeSigConst = ConstantInt::get(intType, nodeSig);
        new StoreInst(nodeSigConst, G, &node->getBasicBlock()->front());
      }
    }
  }

  /* Find the CFCSS designated failure block in the given function */
  BasicBlock *CFCSS::findFailBlock(Function *f) {
    Function::iterator FI = f->begin();
    Function::iterator FE = f->end();

    for (; FI != FE; FI++) {
      if (FI->getName() == failBlockName)
        return FI;
    }
    return NULL;
  }

  /* Gets the set of vertices this block has outgoing
   * connections to */
  std::set<BasicBlock *> CFCSS::getToVertices(BasicBlock *block) {
    std::set<BasicBlock *> results;

    std::set<CFGraphEdge>::iterator iter = this->edges->begin();
    std::set<CFGraphEdge>::iterator end = this->edges->end();

    for (; iter != end; iter++) {
      if (iter->getFrom() == block)
        results.insert(iter->getTo());
    }

    return results;
  }

  /* Gets the set of vertices this block has incoming
   * connections from */
  std::set<BasicBlock *> CFCSS::getFromVertices(BasicBlock *block) {
    std::set<BasicBlock *> results;

    std::set<CFGraphEdge>::iterator iter = this->edges->begin();
    std::set<CFGraphEdge>::iterator end = this->edges->end();

    for (; iter != end; iter++) {
      if (iter->getTo() == block)
        results.insert(iter->getFrom());
    }

    return results;
  }

  const std::string CFCSS::failBlockName = "cfcssFail";
}

char CFCSS::ID = 0;
INITIALIZE_PASS(CFCSS, 
                "cfcss",
                "Control flow checking via software signatures",
                false,
                false)

ModulePass *llvm::createCFCSSPass() {
  return new CFCSS();
}

