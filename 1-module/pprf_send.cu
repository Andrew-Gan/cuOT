#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aesExpand.h"
#include "base_ot.h"

using KeyPair = std::pair<unsigned*, unsigned*>;

__host__
TreeNode* worker_sender(TreeNode root, KeyPair keys, int tid, int treeStart, int treeEnd, int depth) {
  BaseOT baseOT(Sender, tid);
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  TreeNode *input_d, *output_d, *subtotal_d;
  cudaError_t err0 = cudaMalloc(&input_d, sizeof(*input_d) * numLeaves / 2 + PADDED_LEN);
  cudaError_t err1 = cudaMalloc(&output_d, sizeof(*output_d) * numLeaves);
  cudaError_t err2 = cudaMalloc(&subtotal_d, sizeof(*subtotal_d) * numLeaves);
  cudaMemset(subtotal_d, 0, sizeof(*subtotal_d) * numLeaves);

  if (err0 != cudaSuccess)
    fprintf(stderr, "send in: %s\n", cudaGetErrorString(err0));
  if (err1 != cudaSuccess)
    fprintf(stderr, "send out: %s\n", cudaGetErrorString(err1));
  if (err2 != cudaSuccess)
    fprintf(stderr, "send sub: %s\n", cudaGetErrorString(err2));

  for (int t = treeStart; t <= treeEnd; t++) {
    cudaMemcpy(output_d, &root, sizeof(root), cudaMemcpyHostToDevice);

    for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(input_d, output_d, sizeof(*output_d) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*output_d);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / AES_BSIZE, 1);
      dim3 thread(AES_BSIZE, 1);
      AesBlocks m0(width / 2), m1(width / 2);
      aesExpand128<<<grid, thread>>>(keys.first, output_d, (uint32_t*) m0.data_d, (unsigned*) input_d, 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, output_d, (uint32_t*) m1.data_d, (unsigned*) input_d, 1, width);
      cudaDeviceSynchronize();

      baseOT.send(m0, m1);
    }

    xor_prf<<<tBlock, 1024>>>(subtotal_d, output_d, numLeaves);
    cudaDeviceSynchronize();
  }

  cudaFree(input_d);
  cudaFree(output_d);
  return subtotal_d;
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

  TreeNode *fullVec_d;
  cudaError_t err = cudaMalloc(&fullVec_d, sizeof(*fullVec_d) * numLeaves);
  cudaMemset(fullVec_d, 0, sizeof(*fullVec_d) * numLeaves);

  if (err != cudaSuccess)
    fprintf(stderr, "send full: %s\n", cudaGetErrorString(err));

  uint64_t delta = 0;

  int workload = (numTrees - 1) / EXP_NUM_THREAD + 1;
  std::vector<std::future<TreeNode*>> workers;
  KeyPair keys = std::make_pair(leftKey_d, rightKey_d);
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    int treeStart = tid * workload;
    int treeEnd = ((tid+1) * workload - 1);
    if (treeEnd > (numTrees - 1))
      treeEnd = numTrees - 1;
    workers.push_back(std::async(worker_sender, root, keys, tid, treeStart, treeEnd, depth));
  }
  int tBlock = (numLeaves - 1) / 1024 + 1;
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    TreeNode *subtotal_d = workers.at(tid).get();
    xor_prf<<<tBlock, 1024>>>(fullVec_d, subtotal_d, numLeaves);
    cudaDeviceSynchronize();
    cudaFree(subtotal_d);
  }

  cudaFree(leftKey_d);
  cudaFree(rightKey_d);

  Vector fullVec_dtor =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) fullVec_d };

  return {fullVec_dtor, delta};
}
