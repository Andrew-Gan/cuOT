#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aes_expand.h"
#include "simplest_ot.h"
#include "basic_op.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

__global__
void set_choice(SparseVector choiceVector, int index, int t) {
  choiceVector.nonZeros[t] = index;
}

static std::pair<GPUBlock, SparseVector> expander(KeyPair keys, uint64_t *choices, int numTrees, int depth) {
  EventLog::start(BufferInit);
  size_t numLeaves = pow(2, depth);
  GPUBlock input(numTrees * numLeaves * TREENODE_SIZE);
  GPUBlock output(numTrees * numLeaves * TREENODE_SIZE);
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(numLeaves * TREENODE_SIZE / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(numLeaves * TREENODE_SIZE / 2));
  std::vector<std::vector<GPUBlock>> sum(numTrees, std::vector<GPUBlock>(depth+1, GPUBlock(TREENODE_SIZE)));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);
  std::vector<size_t> puncture(numTrees, 0);

  SparseVector choiceVector = {
    .nBits = numLeaves,
  };
  cudaError_t err = cudaMalloc(&choiceVector.nonZeros, numTrees * sizeof(size_t));
  if (err != cudaSuccess)
    fprintf(stderr, "choice vec: %s\n", cudaGetErrorString(err));

  for (int t = 0; t < numTrees; t++) {
    baseOT.push_back(new SimplestOT(OT::Recver, t+1));
  }
  EventLog::end(BufferInit);

  // obtain sums of every layer of every tree
  std::vector<std::future<std::vector<GPUBlock>>> baseOTWorkers;
  for (int t = 0; t < numTrees; t++) {
    baseOTWorkers.push_back(std::async([t, &baseOT, choices]() {
      return baseOT.at(t)->recv(choices[t]);
    }));
  }
  for (int t = 0; t < numTrees; t++) {
    sum.at(t) = baseOTWorkers.at(t).get();
  }

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    EventLog::start(PprfRecverExpand);
    // expand layer
    for (int t = 0; t < numTrees; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * width;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * width;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfRecverExpand);

    // insert obtained sum into layer
    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      TreeNode *sideCasted = (TreeNode*) side->data_d;
      int recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpy(&sideCasted[recvNodeId / 2], sum.at(t).at(d-1).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        sideCasted = (TreeNode*) xorSide->data_d;
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&sideCasted[deltaNodeId / 2], sum.at(t).at(d).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
       }
    }

    // conduct sum/xor in parallel
    EventLog::start(SumNodes);
    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      side->sum_async(TREENODE_SIZE);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        xorSide->sum_async(TREENODE_SIZE);
      }
    }
    cudaDeviceSynchronize();

    // insert active node obtained from sum into output
    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      TreeNode *oCasted = (TreeNode*) output.data_d + t * numLeaves;
      int recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpy(&oCasted[recvNodeId], side->data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);

      if(d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&oCasted[deltaNodeId], xorSide->data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
    EventLog::end(SumNodes);
  }

  for (int t = 0; t < numTrees; t++) {
    set_choice<<<1, 1>>>(choiceVector, t*numLeaves + puncture.at(t), t);
    cudaDeviceSynchronize();
    choiceVector.weight++;
  }

  return std::make_pair(output, choiceVector);
}

std::pair<GPUBlock, SparseVector> pprf_recver(uint64_t *choices, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);

  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  KeyPair keys = std::make_pair(k0_blk, k1_blk);
  auto [puncVector, choiceVector] = expander(keys, choices, numTrees, depth);

  return {puncVector, choiceVector};
}
