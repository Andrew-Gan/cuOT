#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf_gpu.h"
#include "aesExpand.h"

using KeyPair = std::pair<unsigned*, unsigned*>;

__host__
TreeNode* worker_sender(TreeNode root, KeyPair keys, uint64_t *choices, int tid, int treeStart, int treeEnd, int depth) {
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  TreeNode *d_input, *d_output, *d_subtotal;
  cudaError_t err0 = cudaMalloc(&d_input, sizeof(*d_input) * numLeaves / 2 + PADDED_LEN);
  cudaError_t err1 = cudaMalloc(&d_output, sizeof(*d_output) * numLeaves);
  cudaError_t err2 = cudaMalloc(&d_subtotal, sizeof(*d_subtotal) * numLeaves);
  cudaMemset(d_subtotal, 0, sizeof(*d_subtotal) * numLeaves);

#ifdef DEBUG_MODE
  if (err0 != cudaSuccess) printf("send in: %s\n", cudaGetErrorString(err0));
  if (err1 != cudaSuccess) printf("send out: %s\n", cudaGetErrorString(err1));
  if (err2 != cudaSuccess) printf("send sub: %s\n", cudaGetErrorString(err2));
#endif

  for (int t = treeStart; t <= treeEnd; t++) {
    TreeNode *tmp;
    cudaMalloc(&tmp, sizeof(*d_otNodes[t]) * depth);
    d_otNodes[t] = tmp;

    int puncture = 0;
    cudaMemcpy(d_output, &root, sizeof(root), cudaMemcpyHostToDevice);

    for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_input, d_output, sizeof(*d_output) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_output);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / AES_BSIZE, 1);
      dim3 thread(AES_BSIZE, 1);
      aesExpand128<<<grid, thread>>>(keys.first, d_output,  (unsigned*) d_input, 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, d_output,  (unsigned*) d_input, 1, width);
      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncture * 2 + 1 - (width - 1) + choice;
      cudaMemcpy(&d_otNodes[t][d-1], &d_output[otLeafLayerIdx], sizeof(*d_output), cudaMemcpyDeviceToDevice);
      puncture = puncture * 2 + 1 + (1 - choice);
    }

    treeExpanded[t] = true;
    xor_prf<<<tBlock, 1024>>>(d_subtotal, d_output, numLeaves);
    cudaDeviceSynchronize();
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return d_subtotal;
}

std::pair<Vector, uint64_t> pprf_sender_gpu(uint64_t *choices, TreeNode root, int depth, int numTrees) {
  size_t numLeaves = pow(2, depth);

  // keys to use for tree expansion
  AES_ctx leftAesKey, rightAesKey;
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k_blk[16] = {0};
  unsigned *d_leftKey, *d_rightKey;

  memcpy(&k_blk[8], &k0, sizeof(k0));
  Aes::expand_key(leftAesKey.roundKey, k_blk);
  cudaMalloc(&d_leftKey, sizeof(leftAesKey));
  cudaMemcpy(d_leftKey, &leftAesKey, sizeof(leftAesKey), cudaMemcpyHostToDevice);
  memset(&k_blk, 0, sizeof(k_blk));

  memcpy(&k_blk[8], &k1, sizeof(k1));
  Aes::expand_key(rightAesKey.roundKey, k_blk);
  cudaMalloc(&d_rightKey, sizeof(rightAesKey));
  cudaMemcpy(d_rightKey, &rightAesKey, sizeof(rightAesKey), cudaMemcpyHostToDevice);

  TreeNode *d_fullVec;
  cudaError_t err = cudaMalloc(&d_fullVec, sizeof(*d_fullVec) * numLeaves);
  cudaMemset(d_fullVec, 0, sizeof(*d_fullVec) * numLeaves);

#ifdef DEBUG_MODE
  if (err != cudaSuccess) printf("send full: %s\n", cudaGetErrorString(err));
#endif

  uint64_t delta = 0;
  d_otNodes = new std::atomic<TreeNode*>[numTrees];
  treeExpanded = new std::atomic<bool>[numTrees]();

  int workload = (numTrees - 1) / EXP_NUM_THREAD + 1;
  std::vector<std::future<TreeNode*>> workers;
  KeyPair keys = std::make_pair(d_leftKey, d_rightKey);
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    int treeStart = tid * workload;
    int treeEnd = ((tid+1) * workload - 1);
    if (treeEnd > (numTrees - 1))
      treeEnd = numTrees - 1;
    workers.push_back(std::async(worker_sender, root, keys, choices, tid, treeStart, treeEnd, depth));
  }
  int tBlock = (numLeaves - 1) / 1024 + 1;
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    TreeNode *d_subtotal = workers.at(tid).get();
    xor_prf<<<tBlock, 1024>>>(d_fullVec, d_subtotal, numLeaves);
    cudaDeviceSynchronize();
    cudaFree(d_subtotal);
  }

  cudaFree(d_leftKey);
  cudaFree(d_rightKey);

  Vector d_fullVector =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) d_fullVec };

  return {d_fullVector, delta};
}
