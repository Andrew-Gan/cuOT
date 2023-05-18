#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aes_expand.h"
#include "simplest_ot.h"

using KeyPair = std::pair<unsigned*, unsigned*>;

__global__
void set_choice(Vector choiceVec, int index) {
  if (index >= choiceVec.n) {
    return;
  }
  choiceVec.data[index / 8] |= 1 << (index % 8);
}

__host__
TreeNode* worker_recver(Vector choiceVector, KeyPair keys, uint64_t *choices, int numTrees, int depth) {
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  std::vector<TreeNode*> input_d(numTrees);
  std::vector<TreeNode*> output_d(numTrees);
  std::vector<SimplestOT*> baseOT;
  std::vector<int> puncture(numTrees, 0);
  TreeNode *puncVector;

  cudaMalloc(&puncVector, sizeof(*puncVector) * numLeaves);
  cudaMemset(puncVector, 0, sizeof(*puncVector) * numLeaves);

  for (int t = 0; t < numTrees; t++) {
    cudaMalloc(&input_d.at(t), sizeof(*input_d.at(t)) * numLeaves / 2 + PADDED_LEN);
    cudaMalloc(&output_d.at(t), sizeof(*output_d.at(t)) * numLeaves);
    baseOT.push_back(new SimplestOT(Recver, t));
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

    EventLog::start(PprfRecverExpand);
    for (int t = 0; t < numTrees; t++) {
      aesExpand128<<<grid, thread>>>(keys.first, output_d.at(t), nullptr, (uint32_t*) input_d.at(t), 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, output_d.at(t), nullptr, (uint32_t*) input_d.at(t), 1, width);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfRecverExpand);

    for (int t = 0; t < numTrees; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int recvNode = puncture.at(t) * 2 + choice;
      GPUBlock mb = baseOT.at(t)->recv(choice);
      // cuda-memcheck returns error due to copying from another context
      cudaMemcpy(&output_d[recvNode], &mb.data_d[puncture.at(t)*TREENODE_SIZE], TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      puncture.at(t) = puncture.at(t) * 2 + (1 - choice);
    }
  }

  for (int t = 0; t < numTrees; t++) {
    xor_prf<<<tBlock, 1024>>>(puncVector, output_d.at(t), numLeaves);
    set_choice<<<1, 1>>>(choiceVector, puncture.at(t));
    cudaDeviceSynchronize();
    cudaFree(input_d.at(t));
    cudaFree(output_d.at(t));
  }
  return puncVector;
}

std::pair<Vector, Vector> pprf_recver(uint64_t *choices, int depth, int numTrees) {
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

  Vector choiceVector = { .n = numLeaves * 8 * TREENODE_SIZE };
  cudaError_t err = cudaMalloc(&choiceVector.data, numLeaves * TREENODE_SIZE);
  if (err != cudaSuccess)
    fprintf(stderr, "recv choice: %s\n", cudaGetErrorString(err));

  cudaMemset(choiceVector.data, 0, numLeaves * TREENODE_SIZE);


  KeyPair keys = std::make_pair(leftKey_d, rightKey_d);
  TreeNode *puncVector = worker_recver(choiceVector, keys, choices, numTrees, depth);

  cudaFree(leftKey_d);
  cudaFree(rightKey_d);

  Vector puncVec =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) puncVector };

  return {puncVec, choiceVector};
}
