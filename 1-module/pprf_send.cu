#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "simplest_ot.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

TreeNode* expander(TreeNode root, KeyPair keys, int numTrees, int depth) {
  int numLeaves = pow(2, depth);
  EventLog::start(BufferInit);
  std::vector<TreeNode*> input_d(numTrees);
  std::vector<TreeNode*> output_d(numTrees);
  size_t blockSize = numLeaves < 1024 ? 1024 : numLeaves;
  std::vector<GPUBlock> m0(numTrees, GPUBlock(blockSize));
  std::vector<GPUBlock> m1(numTrees, GPUBlock(blockSize));
  std::vector<SimplestOT*> baseOT;
  TreeNode *fullVector;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);

  cudaMalloc(&fullVector, sizeof(*fullVector) * numLeaves);
  cudaMemset(fullVector, 0, sizeof(*fullVector) * numLeaves);

  for (int t = 0; t < numTrees; t++) {
    cudaMalloc(&input_d.at(t), sizeof(*input_d.at(t)) * numLeaves / 2 + PADDED_LEN);
    cudaMalloc(&output_d.at(t), sizeof(*output_d.at(t)) * numLeaves);
    baseOT.push_back(new SimplestOT(Sender, t));
    cudaMemcpy(output_d.at(t), &root, sizeof(root), cudaMemcpyHostToDevice);
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (int t = 0; t < numTrees; t++) {
      cudaMemcpy(input_d.at(t), output_d.at(t), sizeof(*output_d.at(t)) * width / 2, cudaMemcpyDeviceToDevice);
    }

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < numTrees; t++) {
      aesLeft.hash_async(output_d.at(t), m0.at(t), input_d.at(t), width, 0);
      aesRight.hash_async(output_d.at(t), m1.at(t), input_d.at(t), width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfSenderExpand);

    std::vector<std::future<void>> baseOTWorkers;
    for (int t = 0; t < numTrees; t++) {
      baseOTWorkers.push_back(std::async([t, &baseOT, &m0, &m1]() {
        baseOT.at(t)->send(m0.at(t), m1.at(t));
      }));
    }
    for (std::future<void> &worker : baseOTWorkers) {
      worker.get();
    }
  }

  int tBlock = (numLeaves - 1) / 1024 + 1;
  for (int t = 0; t < numTrees; t++) {
    xor_prf<<<tBlock, 1024>>>(fullVector, output_d.at(t), numLeaves);
    cudaDeviceSynchronize();
    cudaFree(input_d.at(t));
    cudaFree(output_d.at(t));
  }
  return fullVector;
}

std::pair<Vector, uint64_t> pprf_sender(TreeNode root, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);
  uint64_t delta = 0;
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  KeyPair keys = std::make_pair(k0_blk, k1_blk);
  TreeNode *fullVector = expander(root, keys, numTrees, depth);

  Vector fullVec =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) fullVector };

  return {fullVec, delta};
}
