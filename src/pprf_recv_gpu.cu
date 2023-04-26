#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf_gpu.h"
#include "aesExpand_kernel.h"
#include "aesCudaUtils.hpp"

using KeyPair = std::pair<unsigned*, unsigned*>;

__global__
void set_choice(Vector choiceVec, int index) {
  if (index >= choiceVec.n) {
    return;
  }
  choiceVec.data[index / 8] |= 1 << (index % 8);
}

__host__
TreeNode* worker_recver(Vector d_choiceVector, KeyPair keys, uint64_t *choices, int tid, int treeStart, int treeEnd, int depth) {
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  TreeNode *d_input, *d_output, *d_subtotal;
  cudaError_t err0 = cudaMalloc(&d_input, sizeof(*d_input) * numLeaves / 2 + PADDED_LEN);
  cudaError_t err1 = cudaMalloc(&d_output, sizeof(*d_output) * numLeaves);
  cudaError_t err2 = cudaMalloc(&d_subtotal, sizeof(*d_subtotal) * numLeaves);
  cudaMemset(d_subtotal, 0, sizeof(*d_subtotal) * numLeaves);

#ifdef DEBUG_MODE
  if (err0 != cudaSuccess) printf("recv in: %s\n", cudaGetErrorString(err));
  if (err1 != cudaSuccess) printf("recv out: %s\n", cudaGetErrorString(err));
  if (err2 != cudaSuccess) printf("recv sub: %s\n", cudaGetErrorString(err));
#endif

  for (int t = treeStart; t <= treeEnd; t++) {
    while (!treeExpanded[t]);
    int choice = choices[t] & 1;
    int puncture = 1 - choice;
    cudaMemcpy(&d_output[choice], &d_otNodes[t][0], sizeof(*d_otNodes[t]), cudaMemcpyDeviceToDevice);

    for (size_t d = 2, width = 4; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_input, d_output, sizeof(*d_output) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_output);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(keys.first, d_output, (unsigned*) d_input, 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, d_output, (unsigned*) d_input, 1, width);
      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncture * 2 + choice;
      cudaMemcpy(&d_output[otLeafLayerIdx], &d_otNodes[t][d-1], sizeof(*d_otNodes[t]), cudaMemcpyDeviceToDevice);
      puncture = puncture * 2 + (1 - choice);
    }

    xor_prf<<<tBlock, 1024>>>(d_subtotal, d_output, numLeaves);
    set_choice<<<1, 1>>>(d_choiceVector, puncture);
    cudaDeviceSynchronize();
    cudaFree(d_otNodes[t]);
  }

  cudaFree(d_input);
  cudaFree(d_output);
  return d_subtotal;
}

std::pair<Vector, Vector> pprf_recver_gpu(uint64_t *choices, int depth, int numTrees) {
  cuda_check();
  size_t numLeaves = pow(2, depth);

  // keys to use for tree expansion
  AES_ctx leftAesKey, rightAesKey;
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k_blk[16] = {0};
  unsigned *d_leftKey, *d_rightKey;

  memcpy(&k_blk[8], &k0, sizeof(k0));
  aes_init_ctx(&leftAesKey, k_blk);
  cudaMalloc(&d_leftKey, sizeof(leftAesKey));
  cudaMemcpy(d_leftKey, &leftAesKey, sizeof(leftAesKey), cudaMemcpyHostToDevice);
  memset(&k_blk, 0, sizeof(k_blk));

  memcpy(&k_blk[8], &k1, sizeof(k1));
  aes_init_ctx(&rightAesKey, k_blk);
  cudaMalloc(&d_rightKey, sizeof(rightAesKey));
  cudaMemcpy(d_rightKey, &rightAesKey, sizeof(rightAesKey), cudaMemcpyHostToDevice);

  // store tree in device memory
  TreeNode *d_puncVec;
  cudaError_t err0 = cudaMalloc(&d_puncVec, numLeaves * sizeof(*d_puncVec));
  cudaMemset(d_puncVec, 0, numLeaves * sizeof(*d_puncVec));

  Vector d_choiceVector;
  cudaError_t err1 = cudaMalloc(&d_choiceVector.data, numLeaves * sizeof(*d_puncVec));
  cudaMemset(d_choiceVector.data, 0, numLeaves * sizeof(*d_puncVec));

#ifdef DEBUG_MODE
  if (err0 != cudaSuccess) printf("recv punc: %s\n", cudaGetErrorString(err));
  if (err1 != cudaSuccess) printf("recv choice: %s\n", cudaGetErrorString(err));
#endif

  int workload = (numTrees - 1) / EXP_NUM_THREAD + 1;
  std::vector<std::future<TreeNode*>> workers;
  KeyPair keys = std::make_pair(d_leftKey, d_rightKey);

  while (treeExpanded == nullptr);
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    int treeStart = tid * workload;
    int treeEnd = ((tid+1) * workload - 1);
    if (treeEnd > (numTrees - 1))
      treeEnd = numTrees - 1;
    workers.push_back(std::async(worker_recver, d_choiceVector, keys, choices, tid, treeStart, treeEnd, depth));
  }
  int tBlock = (numLeaves - 1) / 1024 + 1;
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    TreeNode *d_subtotal = workers.at(tid).get();
    xor_prf<<<tBlock, 1024>>>(d_puncVec, d_subtotal, numLeaves);
    cudaDeviceSynchronize();
    cudaFree(d_subtotal);
  }

  delete[] d_otNodes;
  delete[] treeExpanded;
  cudaFree(d_leftKey);
  cudaFree(d_rightKey);

  Vector d_puncVector =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) d_puncVec };

  return {d_puncVector, d_choiceVector};
}
