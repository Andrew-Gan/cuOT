#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "simplest_ot.h"
#include "basic_op.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

static std::pair<GPUBlock, GPUBlock> expander(TreeNode root, KeyPair keys, int numTrees, int depth) {
  EventLog::start(BufferInit);
  GPUBlock delta(TREENODE_SIZE);
  delta.set(123456);
  size_t numLeaves = pow(2, depth);
  size_t bufferSize = numTrees * numLeaves * TREENODE_SIZE;
  GPUBlock input(bufferSize);
  GPUBlock output(bufferSize);
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<std::vector<GPUBlock>> leftSum(numTrees, std::vector<GPUBlock>(depth+1, GPUBlock(TREENODE_SIZE)));
  std::vector<std::vector<GPUBlock>> rightSum(numTrees, std::vector<GPUBlock>(depth+1, GPUBlock(TREENODE_SIZE)));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);

  for (int t = 0; t < numTrees; t++) {
    baseOT.push_back(new SimplestOT(Sender, t));
    outputs.at(t).set((uint8_t*) root.data, TREENODE_SIZE);
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < numTrees; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * width;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * width;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfSenderExpand);

    for (int t = 0; t < numTrees; t++) {
      leftSum.at(t).at(d).minCopy(leftNodes.at(t).sum(TREENODE_SIZE));
      rightSum.at(t).at(d).minCopy(rightNodes.at(t).sum(TREENODE_SIZE));
    }
    cudaDeviceSynchronize();

    if (d == depth) {
      for (int t = 0; t < numTrees; t++) {
        leftSum.at(t).at(d) = leftSum.at(t).at(d-1);
        m0XorDelta ^= delta;
        rightSum.at(t).at(d) = rightSum.at(t).at(d-1);
        m1XorDelta ^= delta;
      }
    }
  }

  std::vector<std::future<void>> baseOTWorkers;
  for (int t = 0; t < numTrees; t++) {
    baseOTWorkers.push_back(std::async([t, d, &baseOT, &leftSum, &rightSum]() {
      baseOT.at(t)->send(leftSum.at(t), rightSum.at(t));
    }));
  }
  for (std::future<void> &worker : baseOTWorkers) {
    worker.get();
  }

  return std::make_pair(output, delta);
}

std::pair<GPUBlock, GPUBlock> pprf_sender(TreeNode root, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  KeyPair keys = std::make_pair(k0_blk, k1_blk);
  auto [fullVector, delta] = expander(root, keys, numTrees, depth);

  return {fullVector, delta};
}
