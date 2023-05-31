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
  std::vector<GPUBlock> inputs(numTrees, GPUBlock(bufferSize));
  std::vector<GPUBlock> outputs(numTrees, GPUBlock(bufferSize));
  std::vector<GPUBlock> leftNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> rightNodes(numTrees, GPUBlock(bufferSize / 2));
  std::vector<GPUBlock> recvNode(numTrees, GPUBlock(TREENODE_SIZE));
  std::vector<GPUBlock> deltaNode(numTrees, GPUBlock(TREENODE_SIZE));
  std::vector<GPUBlock> sum(numTrees, GPUBlock(2 * TREENODE_SIZE));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(keys.first);
  Aes aesRight(keys.second);
  std::vector<size_t> puncture(numTrees, 0);
  GPUBlock puncVector(bufferSize);
  puncVector.set(0);

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

  EventLog::start(PprfRecverExpand);
  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (int t = 0; t < numTrees; t++) {
      inputs.at(t) = outputs.at(t);
    }

    for (int t = 0; t < numTrees; t++) {
      aesLeft.expand_async((TreeNode*) outputs.at(t).data_d, leftNodes.at(t), (TreeNode*) inputs.at(t).data_d, width, 0);
      aesRight.expand_async((TreeNode*) outputs.at(t).data_d, rightNodes.at(t), (TreeNode*) inputs.at(t).data_d, width, 1);
    }
    cudaDeviceSynchronize();

    std::vector<std::future<GPUBlock>> baseOTWorkers;
    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      baseOTWorkers.push_back(std::async([t, &baseOT, choice]() {
        return baseOT.at(t)->recv(choice);
      }));
    }

    for (int t = 0; t < numTrees; t++) {
      sum.at(t) = baseOTWorkers.at(t).get();
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int recvNodeId = puncture.at(t) * 2 + choice;
      TreeNode *oCasted = nullptr;

      if (choice == 0) {
        oCasted = (TreeNode*) leftNodes.at(t).data_d;
        cudaMemcpy(&oCasted[recvNodeId / 2], sum.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
        recvNode.at(t) = leftNodes.at(t).sum(TREENODE_SIZE);
      }
      else {
        oCasted = (TreeNode*) rightNodes.at(t).data_d;
        cudaMemcpy(&oCasted[recvNodeId / 2], sum.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
        recvNode.at(t) = rightNodes.at(t).sum(TREENODE_SIZE);
      }

      if (d == depth) {
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        if (choice == 0) {
          oCasted = (TreeNode*) rightNodes.at(t).data_d;
          cudaMemcpy(&oCasted[deltaNodeId / 2], sum.at(t).data_d + TREENODE_SIZE, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
          deltaNode.at(t) = rightNodes.at(t).sum(TREENODE_SIZE);
        }
        else {
          oCasted = (TreeNode*) leftNodes.at(t).data_d;
          cudaMemcpy(&oCasted[deltaNodeId / 2], sum.at(t).data_d + TREENODE_SIZE, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
          deltaNode.at(t) = leftNodes.at(t).sum(TREENODE_SIZE);
        }
      }
    }

    for (int t = 0; t < numTrees; t++) {
      TreeNode *oCasted = (TreeNode*) outputs.at(t).data_d;
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      size_t recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpy(&oCasted[recvNodeId], recvNode.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);

      if (d == depth) {
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&oCasted[deltaNodeId], deltaNode.at(t).data_d, TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
  }

  for (int t = 0; t < numTrees; t++) {
    puncVector.append(outputs.at(t));
    set_choice<<<1, 1>>>(choiceVector, t*numLeaves + puncture.at(t), t);
    cudaDeviceSynchronize();
    choiceVector.weight++;
  }
  EventLog::end(PprfRecverExpand);

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
