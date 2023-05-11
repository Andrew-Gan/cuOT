#include <atomic>
#include <vector>
#include <future>

#include "aes.h"
#include "pprf.h"
#include "aesExpand.h"
#include "base_ot.h"

using KeyPair = std::pair<unsigned*, unsigned*>;

__global__
void set_choice(Vector choiceVec, int index) {
  if (index >= choiceVec.n) {
    return;
  }
  choiceVec.data[index / 8] |= 1 << (index % 8);
}

__host__
TreeNode* worker_recver(Vector choiceVector_d, KeyPair keys, uint64_t *choices, int tid, int treeStart, int treeEnd, int depth) {
  BaseOT baseOT(Recver, tid);
  int numLeaves = pow(2, depth);
  int tBlock = (numLeaves - 1) / 1024 + 1;
  TreeNode *input_d, *output_d, *subTotal_d;
  cudaError_t err0 = cudaMalloc(&input_d, sizeof(*input_d) * numLeaves / 2 + PADDED_LEN);
  cudaError_t err1 = cudaMalloc(&output_d, sizeof(*output_d) * numLeaves);
  cudaError_t err2 = cudaMalloc(&subTotal_d, sizeof(*subTotal_d) * numLeaves);
  cudaMemset(subTotal_d, 0, sizeof(*subTotal_d) * numLeaves);

  if (err0 != cudaSuccess)
    fprintf(stderr, "recv in: %s\n", cudaGetErrorString(err0));
  if (err1 != cudaSuccess)
    fprintf(stderr, "recv out: %s\n", cudaGetErrorString(err1));
  if (err2 != cudaSuccess)
    fprintf(stderr, "recv sub: %s\n", cudaGetErrorString(err2));

  for (int t = treeStart; t <= treeEnd; t++) {
    int puncture = 0, width = 2;
    for (size_t d = 1; d <= depth; d++) {

      // copy previous layer for expansion
      cudaMemcpy(input_d, output_d, sizeof(*output_d) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*output_d);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / AES_BSIZE, 1);
      dim3 thread(AES_BSIZE, 1);
      aesExpand128<<<grid, thread>>>(keys.first, output_d, nullptr, (unsigned*) input_d, 0, width);
      aesExpand128<<<grid, thread>>>(keys.second, output_d, nullptr, (unsigned*) input_d, 1, width);
      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int recvNode = puncture * 2 + choice;
      GPUBlock mb = baseOT.recv(choice);
      cudaMemcpy(&output_d[recvNode], mb[puncture], TREENODE_SIZE, cudaMemcpyDeviceToDevice);
      puncture = puncture * 2 + (1 - choice);

      width *= 2;
    }

    xor_prf<<<tBlock, 1024>>>(subTotal_d, output_d, numLeaves);
    set_choice<<<1, 1>>>(choiceVector_d, puncture);
    cudaDeviceSynchronize();
  }

  cudaFree(input_d);
  cudaFree(output_d);
  return subTotal_d;
}

std::pair<Vector, Vector> pprf_recver(uint64_t *choices, int depth, int numTrees) {
  EventLog::start(PprfRecver);
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

  // store tree in device memory
  TreeNode *puncVec_d;
  cudaError_t err0 = cudaMalloc(&puncVec_d, numLeaves * sizeof(*puncVec_d));
  cudaMemset(puncVec_d, 0, numLeaves * sizeof(*puncVec_d));

  Vector choiceVector_d;
  cudaError_t err1 = cudaMalloc(&choiceVector_d.data, numLeaves * sizeof(*puncVec_d));
  cudaMemset(choiceVector_d.data, 0, numLeaves * sizeof(*puncVec_d));

  if (err0 != cudaSuccess)
    fprintf(stderr, "recv punc: %s\n", cudaGetErrorString(err0));
  if (err1 != cudaSuccess)
    fprintf(stderr, "recv choice: %s\n", cudaGetErrorString(err1));

  int workload = (numTrees - 1) / EXP_NUM_THREAD + 1;
  std::vector<std::future<TreeNode*>> workers;
  KeyPair keys = std::make_pair(leftKey_d, rightKey_d);

  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    int treeStart = tid * workload;
    int treeEnd = ((tid+1) * workload - 1);
    if (treeEnd > (numTrees - 1))
      treeEnd = numTrees - 1;
    workers.push_back(std::async(worker_recver, choiceVector_d, keys, choices, tid, treeStart, treeEnd, depth));
  }
  int tBlock = (numLeaves - 1) / 1024 + 1;
  for (int tid = 0; tid < EXP_NUM_THREAD; tid++) {
    TreeNode *subTotal_d = workers.at(tid).get();
    xor_prf<<<tBlock, 1024>>>(puncVec_d, subTotal_d, numLeaves);
    cudaDeviceSynchronize();
    cudaFree(subTotal_d);
  }

  cudaFree(leftKey_d);
  cudaFree(rightKey_d);

  Vector puncVec_dtor =
    { .n = numLeaves * TREENODE_SIZE * 8, .data = (uint8_t*) puncVec_d };

  EventLog::end(PprfRecver);
  return {puncVec_dtor, choiceVector_d};
}
