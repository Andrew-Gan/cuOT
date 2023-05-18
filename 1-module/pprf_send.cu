#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aes_expand.h"
#include "simplest_ot.h"

using KeyPair = std::pair<uint32_t*, uint32_t*>;

__host__
TreeNode* worker_sender(TreeNode root, KeyPair keys, int numTrees, int depth) {
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  std::vector<TreeNode*> input_d(numTrees);
  std::vector<TreeNode*> output_d(numTrees);
  size_t blockSize = numLeaves < 1024 ? 1024 : numLeaves;
  std::vector<GPUBlock> m0(numTrees, GPUBlock(blockSize));
  std::vector<GPUBlock> m1(numTrees, GPUBlock(blockSize));
  std::vector<SimplestOT*> baseOT;
  TreeNode *fullVector;

  cudaMalloc(&fullVector, sizeof(*fullVector) * numLeaves);
  cudaMemset(fullVector, 0, sizeof(*fullVector) * numLeaves);

  for (int t = 0; t < numTrees; t++) {
    cudaMalloc(&input_d.at(t), sizeof(*input_d.at(t)) * numLeaves / 2 + PADDED_LEN);
    cudaMalloc(&output_d.at(t), sizeof(*output_d.at(t)) * numLeaves);
    baseOT.push_back(new SimplestOT(Sender, t));
    cudaMemcpy(output_d.at(t), &root, sizeof(root), cudaMemcpyHostToDevice);
  }

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    // copy previous layer for expansion
    for (int t = 0; t < numTrees; t++) {
      cudaMemcpy(input_d.at(t), output_d.at(t), sizeof(*output_d.at(t)) * width / 2, cudaMemcpyDeviceToDevice);
    }

    size_t paddedLen = (width / 2) * sizeof(*output_d.at(0));
    paddedLen += 16 - (paddedLen % 16);
    paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
    static int thread_per_aesblock = 4;
    dim3 grid(paddedLen * thread_per_aesblock / 16 / AES_BSIZE, 1);
    dim3 thread(AES_BSIZE, 1);

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < numTrees; t++) {
      aesExpand128<<<grid, thread>>>(keys.first, output_d.at(t), (uint32_t*) m0.at(t).data_d, (uint32_t*) input_d.at(t), 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, output_d.at(t), (uint32_t*) m1.at(t).data_d, (uint32_t*) input_d.at(t), 1, width);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfSenderExpand);

    for (int t = 0; t < numTrees; t++) {
      baseOT.at(t)->send(m0.at(t), m1.at(t));
    }
  }

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

  // keys to use for tree expansion
  AES_ctx leftAesKey, rightAesKey;
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k_blk[16] = {0};
  unsigned *leftKey_d, *rightKey_d;

  memcpy(&k_blk[8], &k0, sizeof(k0));
  Aes::expand_encKey(leftAesKey.roundKey, k_blk);
  cudaMalloc(&leftKey_d, sizeof(leftAesKey));
  cudaMemcpy(leftKey_d, &leftAesKey, sizeof(leftAesKey), cudaMemcpyHostToDevice);
  memset(&k_blk, 0, sizeof(k_blk));

  memcpy(&k_blk[8], &k1, sizeof(k1));
  Aes::expand_encKey(rightAesKey.roundKey, k_blk);
  cudaMalloc(&rightKey_d, sizeof(rightAesKey));
  cudaMemcpy(rightKey_d, &rightAesKey, sizeof(rightAesKey), cudaMemcpyHostToDevice);

  uint64_t delta = 0;

  KeyPair keys = std::make_pair(leftKey_d, rightKey_d);
  TreeNode *fullVector = worker_sender(root, keys, numTrees, depth);

  cudaFree(leftKey_d);
  cudaFree(rightKey_d);

  Vector fullVec =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) fullVector };

  return {fullVec, delta};
}
