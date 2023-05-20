#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "simplest_ot.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

GPUBlock expander(TreeNode root, KeyPair keys, int numTrees, int depth) {
  int numLeaves = pow(2, depth);
  EventLog::start(BufferInit);
  std::vector<GPUBlock> inputs(numTrees, GPUBlock(TREENODE_SIZE * numLeaves));
  std::vector<GPUBlock> outputs(numTrees, GPUBlock(TREENODE_SIZE * numLeaves));
  size_t blockSize = numLeaves < 1024 ? 1024 : numLeaves;
  std::vector<GPUBlock> m0(numTrees, GPUBlock(blockSize));
  std::vector<GPUBlock> m1(numTrees, GPUBlock(blockSize));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);
  GPUBlock fullVector(TREENODE_SIZE * numLeaves);

  fullVector.set(0);

  for (int t = 0; t < numTrees; t++) {
    baseOT.push_back(new SimplestOT(Sender, t));
    outputs.at(t).set((uint8_t*) root.data, TREENODE_SIZE);
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (int t = 0; t < numTrees; t++) {
      inputs.at(t) = outputs.at(t);
    }

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < numTrees; t++) {
      aesLeft.hash_async((TreeNode*) outputs.at(t).data_d, m0.at(t), (TreeNode*) inputs.at(t).data_d, width, 0);
      aesRight.hash_async((TreeNode*) outputs.at(t).data_d, m1.at(t), (TreeNode*) inputs.at(t).data_d, width, 1);
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

  for (int t = 0; t < numTrees; t++) {
    fullVector ^= outputs.at(t);
  }
  return fullVector;
}

std::pair<GPUBlock, uint64_t> pprf_sender(TreeNode root, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);
  uint64_t delta = 0;
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  KeyPair keys = std::make_pair(k0_blk, k1_blk);
  GPUBlock fullVector = expander(root, keys, numTrees, depth);

  return {fullVector, delta};
}
