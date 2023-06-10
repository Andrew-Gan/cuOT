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
  size_t bufferSize = std::max(numLeaves * TREENODE_SIZE, (size_t)1024);
  GPUBlock input(bufferSize);
  GPUBlock output(bufferSize);
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> recvNode(numTrees, GPUBlock(TREENODE_SIZE));
  std::vector<GPUBlock> deltaNode(numTrees, GPUBlock(TREENODE_SIZE));
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
    baseOT.push_back(new SimplestOT(Recver, t));
  }
  EventLog::end(BufferInit);

  std::vector<std::future<GPUBlock>> baseOTWorkers;
  for (int t = 0; t < numTrees; t++) {
    baseOTWorkers.push_back(std::async([t, &baseOT, choices]() {
      return baseOT.at(t)->recv(choices[t]);
    }));
  }
  for (int t = 0; t < numTrees; t++) {
    sum.at(t) = baseOTWorkers.at(t).get();
  }

  EventLog::start(PprfRecverExpand);
  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    for (int t = 0; t < numTrees; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * width;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * width;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();

    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int recvNodeId = puncture.at(t) * 2 + choice;
      GPUBlock *block = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      TreeNode *oCasted = (TreeNode*) block->data_d;
      cudaMemcpy(&oCasted[recvNodeId / 2], sum.at(t).at(d).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      recvNode.at(t).minCopy(block->sum(TREENODE_SIZE));

      oCasted = (TreeNode*) outputs.at(t).data_d;
      cudaMemcpy(&oCasted[recvNodeId], recvNode.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);

      if (d == depth) {
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        GPUBlock *block = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        oCasted = (TreeNode*) block->data_d;
        cudaMemcpy(&oCasted[deltaNodeId / 2], sum.at(t).at(d+1).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
        deltaNode.at(t).minCopy(block->sum(TREENODE_SIZE));

        oCasted = (TreeNode*) outputs.at(t).data_d;
        cudaMemcpy(&oCasted[deltaNodeId], deltaNode.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
  }

  for (int t = 0; t < numTrees; t++) {
    set_choice<<<1, 1>>>(choiceVector, t*numLeaves + puncture.at(t), t);
    cudaDeviceSynchronize();
    choiceVector.weight++;
  }
  EventLog::end(PprfRecverExpand);

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
