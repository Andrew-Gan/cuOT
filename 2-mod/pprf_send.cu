#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "simplest_ot.h"
#include "silent_ot.h"
#include "basic_op.h"

using KeyPair = std::pair<uint8_t*, uint8_t*>;

std::pair<GPUBlock, GPUBlock> SilentOT::pprf_send(TreeNode root, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  KeyPair keys = {k0_blk, k1_blk};

  EventLog::start(BufferInit);
  GPUBlock delta(TREENODE_SIZE);
  delta.clear();
  delta.set(123456);
  size_t numLeaves = pow(2, depth);
  GPUBlock input(numTrees * numLeaves * TREENODE_SIZE);
  GPUBlock output(numTrees * numLeaves * TREENODE_SIZE);
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(numLeaves * TREENODE_SIZE / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(numLeaves * TREENODE_SIZE / 2));
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);

  for (int t = 0; t < numTrees; t++) {
    output.set((uint8_t*) root.data, TREENODE_SIZE, t * numLeaves * TREENODE_SIZE);
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < numTrees; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * numLeaves;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * numLeaves;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfSenderExpand);

    EventLog::start(SumNodes);
    for (int t = 0; t < numTrees; t++) {
      leftNodes.at(t).sum_async(TREENODE_SIZE);
      rightNodes.at(t).sum_async(TREENODE_SIZE);
    }
    cudaDeviceSynchronize();
    EventLog::end(SumNodes);

    for (int t = 0; t < numTrees; t++) {
      leftHash.at(t).at(d-1) ^= leftNodes.at(t);
      rightHash.at(t).at(d-1) ^= rightNodes.at(t);
    }
    cudaDeviceSynchronize();

    if (d == depth) {
      for (int t = 0; t < numTrees; t++) {
        leftHash.at(t).at(d) ^= leftNodes.at(t);
        leftHash.at(t).at(d) ^= delta;
        rightHash.at(t).at(d) ^= rightNodes.at(t);
        rightHash.at(t).at(d) ^= delta;
      }
    }
  }

  return {output, delta};
}
