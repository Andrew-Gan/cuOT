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
  size_t bufferSize = std::max(numLeaves * TREENODE_SIZE, (size_t)1024);
  std::vector<GPUBlock> inputs(numTrees, GPUBlock(bufferSize));
  std::vector<GPUBlock> outputs(numTrees, GPUBlock(bufferSize));
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(bufferSize / 2));
  size_t sum = std::max(2 * TREENODE_SIZE, 1024);
  std::vector<GPUBlock> leftSum(numTrees, GPUBlock(sum));
  std::vector<GPUBlock> rightSum(numTrees, GPUBlock(sum));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);
  GPUBlock fullVector(bufferSize);
  fullVector.set(0);

  for (int t = 0; t < numTrees; t++) {
    baseOT.push_back(new SimplestOT(Sender, t));
    outputs.at(t).set((uint8_t*) root.data, TREENODE_SIZE);
  }
  EventLog::end(BufferInit);

  EventLog::start(PprfSenderExpand);
  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (int t = 0; t < numTrees; t++) {
      inputs.at(t) = outputs.at(t);
    }

    for (int t = 0; t < numTrees; t++) {
      aesLeft.expand_async((TreeNode*) outputs.at(t).data_d, leftNodes.at(t), (TreeNode*) inputs.at(t).data_d, width, 0);
      aesRight.expand_async((TreeNode*) outputs.at(t).data_d, rightNodes.at(t), (TreeNode*) inputs.at(t).data_d, width, 1);
    }
    cudaDeviceSynchronize();

    for (int t = 0; t < numTrees; t++) {
      leftSum.at(t) = leftNodes.at(t).sum(TREENODE_SIZE);
      rightSum.at(t) = rightNodes.at(t).sum(TREENODE_SIZE);
    }
    cudaDeviceSynchronize();

    if (d == depth) {
      for (int t = 0; t < numTrees; t++) {
        GPUBlock m0XorDelta = leftSum.at(t) ^ delta;
        GPUBlock m1XorDelta = rightSum.at(t) ^ delta;
        TreeNode *m1Casted = (TreeNode*) rightSum.at(t).data_d;
        cudaMemcpy(leftSum.at(t).data_d + TREENODE_SIZE, m1XorDelta.data_d, m1XorDelta.nBytes / 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(rightSum.at(t).data_d + TREENODE_SIZE, m0XorDelta.data_d, m0XorDelta.nBytes / 2, cudaMemcpyDeviceToDevice);
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
  }

  for (int t = 0; t < numTrees; t++) {
    fullVector.append(outputs.at(t));
  }
  EventLog::end(PprfSenderExpand);

  return std::make_pair(fullVector, delta);
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
