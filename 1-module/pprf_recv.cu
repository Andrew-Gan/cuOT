#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aes_expand.h"
#include "simplest_ot.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

__global__
void set_choice(SparseVector choiceVector, int index, int t) {
  choiceVector.nonZeros[t] = index;
}

static std::pair<GPUBlock, SparseVector> expander(KeyPair keys, uint64_t *choices, int numTrees, int depth) {
  EventLog::start(BufferInit);
  size_t numLeaves = pow(2, depth);
  size_t blockSize = 2 * numLeaves * TREENODE_SIZE;
  if (blockSize < 1024)
    blockSize = 1024;
  std::vector<GPUBlock> inputs(numTrees, GPUBlock(blockSize));
  std::vector<GPUBlock> outputs(numTrees, GPUBlock(blockSize));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);
  std::vector<int> puncture(numTrees, 0);
  GPUBlock puncVector(TREENODE_SIZE * numLeaves);
  puncVector.set(0);

  SparseVector choiceVector = {
    .nBits = numLeaves * 8 * TREENODE_SIZE,
  };
  cudaError_t err = cudaMalloc(&choiceVector.nonZeros, numTrees * sizeof(size_t));
  if (err != cudaSuccess)
    fprintf(stderr, "choice vec: %s\n", cudaGetErrorString(err));
  for (int t = 0; t < numTrees; t++) {
    baseOT.push_back(new SimplestOT(Recver, t));
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (int t = 0; t < numTrees; t++) {
      inputs.at(t) = outputs.at(t);
    }

    EventLog::start(PprfRecverExpand);
    for (int t = 0; t < numTrees; t++) {
      aesLeft.hash_async((TreeNode*) outputs.at(t).data_d, nullptr, (TreeNode*) inputs.at(t).data_d, width, 0);
      aesRight.hash_async((TreeNode*) outputs.at(t).data_d, nullptr, (TreeNode*) inputs.at(t).data_d, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfRecverExpand);

    std::vector<std::future<GPUBlock>> baseOTWorkers;
    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      baseOTWorkers.push_back(std::async([t, &baseOT, choice]() {
        return baseOT.at(t)->recv(choice);
      }));
    }
    for (int t = 0; t < numTrees; t++) {
      // cuda-memcheck returns error due to copying from another context
      GPUBlock mb = baseOTWorkers.at(t).get();
      int p = puncture.at(t);
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int recvNode = puncture.at(t) * 2 + choice;
      TreeNode *iCasted = (TreeNode*) mb.data_d;
      TreeNode *oCasted = (TreeNode*) outputs.at(t).data_d;
      cudaMemcpy(&oCasted[recvNode], &iCasted[p], TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      puncture.at(t) = puncture.at(t) * 2 + (1 - choice);

      if (d == depth) {
        size_t deltaNode = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&oCasted[deltaNode], &iCasted[width + p], TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
  }

  for (int t = 0; t < numTrees; t++) {
    puncVector ^= outputs.at(t);
    set_choice<<<1, 1>>>(choiceVector, puncture.at(t), t);
    cudaDeviceSynchronize();
    choiceVector.weight++;
  }
  return std::make_pair(puncVector, choiceVector);
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
