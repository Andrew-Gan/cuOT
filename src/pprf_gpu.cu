#include <atomic>
#include <vector>

#include "aes.h"
#include "aes_gpu.h"
#include "pprf_gpu.h"
#include "aesExpand_kernel.h"
#include "aesCudaUtils.hpp"

// OT content
static std::atomic<TreeNode*> d_otNodes;
static std::atomic<bool*> treeExpanded;

__global__
void print_nodes(TreeNode *nodes, size_t numLeaves) {
  for(int i = 0; i < numLeaves; i++) {
    printf("node %d: ", i);
    for(int j = 0; j < TREENODE_SIZE / 4; j++) {
      printf("%x ", nodes[i].data[j]);
    }
    printf("\n");
  }
  printf("\n");
}

static void cuda_check() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    fprintf(stderr, "There is no device.\n");
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major >= 1)
      break;
  }
  if (dev == deviceCount)
    fprintf(stderr, "There is no device supporting CUDA.\n");
  else
    cudaSetDevice(dev);
}

__global__
void set_choice(Vector choiceVec, int index) {
  if (index >= choiceVec.n) {
    return;
  }
  choiceVec.data[index / 8] |= 1 << (index % 8);
}

__global__
static void xor_prf(TreeNode *sum, TreeNode *operand, size_t numLeaves) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numLeaves) {
    return;
  }
  for (int i = 0; i < TREENODE_SIZE / 4; i++) {
    sum[idx].data[i] ^= operand[idx].data[i];
  }
}

std::pair<Vector, uint64_t> pprf_sender_gpu(uint64_t *choices, TreeNode root, int depth, int numTrees) {
  cuda_check();

  treeExpanded = (bool*) malloc(numTrees * sizeof(*treeExpanded));
  memset((void*) treeExpanded, (int) false, numTrees);
  size_t numLeaves = pow(2, depth);

  // keys to use for tree expansion
  AES_ctx aesKeys[2];
  uint64_t k0 = 3242342;
  uint8_t k0_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  aes_init_ctx(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16] = {0};
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aes_init_ctx(&aesKeys[1], k1_blk);

  cudaResourceDesc resDescLeft;
  cudaResourceDesc resDescRight;
  cudaTextureDesc texDesc;

  // store key in texture memory
  cudaTextureObject_t texLKey = alloc_key_texture(&aesKeys[0], &resDescLeft, &texDesc);
  cudaTextureObject_t texRKey = alloc_key_texture(&aesKeys[1], &resDescRight, &texDesc);

  TreeNode* d_prf;
  cudaMalloc(&d_prf, sizeof(*d_prf) * numLeaves);
  TreeNode* tmp;
  cudaMalloc(&tmp, sizeof(*d_otNodes) * depth);
  d_otNodes = tmp;

  TreeNode *d_InputBuf;
  cudaMalloc(&d_InputBuf, sizeof(*d_InputBuf) * numLeaves / 2 + PADDED_LEN);

  // for storing the accumulated distributed-pd_prf
  TreeNode *d_fullVec;
  cudaMalloc(&d_fullVec, sizeof(*d_fullVec) * numLeaves);
  cudaMemset(d_fullVec, 0, sizeof(*d_fullVec) * numLeaves);

  uint64_t delta = 0;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int t = 0; t < numTrees; t++) {
    int puncture = 0;
    cudaMemcpy(d_prf, &root, sizeof(root), cudaMemcpyHostToDevice);

    for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_InputBuf, d_prf, sizeof(*d_prf) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_prf);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(texLKey, d_prf,  (unsigned*) d_InputBuf, 0, width);
      aesExpand128<<<grid, thread>>>(texRKey, d_prf,  (unsigned*) d_InputBuf, 1, width);
      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncture * 2 + 1 - (width - 1) + choice;
      cudaMemcpy(&d_otNodes[d-1], &d_prf[otLeafLayerIdx], sizeof(*d_prf), cudaMemcpyDeviceToDevice);
      puncture = puncture * 2 + 1 + (1 - choice);
    }

    treeExpanded[t] = true;
    int tBlock = (numLeaves - 1) / 1024 + 1;
    xor_prf<<<tBlock, 1024>>>(d_fullVec, d_prf, numLeaves);
    cudaDeviceSynchronize();
    while(treeExpanded[t] == true);
  }

  free(treeExpanded);
  cudaFree(d_otNodes);
  cudaFree(d_prf);
  cudaFree(d_InputBuf);

  dealloc_key_texture(texLKey);
  dealloc_key_texture(texRKey);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESGPU sender: %0.4f ms\n", duration / NUM_SAMPLES);

  Vector d_fullVector = {
    .n = numLeaves * TREENODE_SIZE * 8,
    .data = (uint8_t*) d_fullVec
  };
  return {d_fullVector, delta};
}

std::pair<Vector, Vector> pprf_recver_gpu(uint64_t *choices, int depth, int numTrees) {
  cuda_check();
  size_t numLeaves = pow(2, depth);

  // keys to use for tree expansion
  AES_ctx aesKeys[2];
  uint64_t k0 = 3242342;
  uint8_t k0_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  aes_init_ctx(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16] = {0};
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aes_init_ctx(&aesKeys[1], k1_blk);

  cudaResourceDesc resDescLeft;
  cudaResourceDesc resDescRight;
  cudaTextureDesc texDesc;

  // store key in texture memory
  cudaTextureObject_t texLKey = alloc_key_texture(&aesKeys[0], &resDescLeft, &texDesc);
  cudaTextureObject_t texRKey = alloc_key_texture(&aesKeys[1], &resDescRight, &texDesc);

  while(treeExpanded == nullptr);

  // store tree in device memory
  TreeNode *d_puncVec;
  cudaMalloc(&d_puncVec, numLeaves * sizeof(*d_puncVec));
  cudaMemset(d_puncVec, 0, numLeaves * sizeof(*d_puncVec));

  Vector d_choiceVector;
  cudaMalloc(&d_choiceVector.data, numLeaves * sizeof(*d_puncVec));
  cudaMemset(d_choiceVector.data, 0, numLeaves * sizeof(*d_puncVec));

  TreeNode *d_pprf;
  cudaMalloc(&d_pprf, numLeaves * sizeof(*d_pprf));
  TreeNode *d_InputBuf;
  cudaMalloc(&d_InputBuf, (numLeaves / 2 + PADDED_LEN) * sizeof(*d_InputBuf));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int t = 0; t < numTrees; t++) {
    while (!treeExpanded[t]);
    int choice = choices[t] & 1;
    int puncture = 1 - choice;
    cudaMemcpy(&d_pprf[choice], &d_otNodes[0], sizeof(*d_otNodes), cudaMemcpyDeviceToDevice);

    for (size_t d = 2, width = 4; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_InputBuf, d_pprf, sizeof(*d_pprf) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_pprf);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(texLKey, d_pprf, (unsigned*) d_InputBuf, 0, width);
      aesExpand128<<<grid, thread>>>(texRKey, d_pprf, (unsigned*) d_InputBuf, 1, width);
      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncture * 2 + choice;
      cudaMemcpy(&d_pprf[otLeafLayerIdx], &d_otNodes[d-1], sizeof(*d_otNodes), cudaMemcpyDeviceToDevice);
      puncture = puncture * 2 + (1 - choice);
    }

    int tBlock = (numLeaves - 1) / 1024 + 1;
    xor_prf<<<tBlock, 1024>>>(d_puncVec, d_pprf, numLeaves);
    set_choice<<<1, 1>>>(d_choiceVector, puncture);
    cudaDeviceSynchronize();
    treeExpanded[t] = false;
  }

  cudaFree(d_pprf);
  cudaFree(d_InputBuf);

  dealloc_key_texture(texLKey);
  dealloc_key_texture(texRKey);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESGPU recver: %0.4f ms\n", duration / NUM_SAMPLES);

  Vector d_puncVector = {
    .n = numLeaves * TREENODE_SIZE,
    .data = (uint8_t*) d_puncVec
  };

  return {d_puncVector, d_choiceVector};
}
