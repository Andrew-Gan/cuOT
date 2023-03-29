#include "aes.h"
#include "aes_gpu.h"
#include "pprf_gpu.h"
#include "aesExpand_kernel.h"
#include "aesCudaUtils.hpp"

// OT content
uint32_t choices[8] = {
  0b01111110011011100010000000111011,
  0b00101011101100101010011001110010,
  0b10110000110000100001110011100100,
  0b00100110101111000000011111011101,
  0b11001000111100000001000111010100,
  0b00111010001111010100011110110101,
  0b11001000111010111100110101100101,
  0b10100001111101000000110011000000,
};

TreeNode *d_otNodes;
volatile bool otSent = false;
TreeNode *d_puncturedLeaves;

__host__
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
void sum_pprf_into_sparse(TreeNode *d_sparse_vec, TreeNode *d_prf,
  TreeNode *d_pprf, TreeNode puncture, size_t numLeaves) {
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numLeaves) {
    for (int i = 0; i < TREENODE_SIZE / 4; i++) {
      // sender prf ^ recver pprf = unit vector
      // unit vector + unit vector + ... = sparse vector
      d_sparse_vec[idx].data[i] ^= d_prf[idx].data[i] ^ d_pprf[idx].data[i];
    }
  }
}

void pprf_sender_gpu(TreeNode *root, TreeNode *leaves, size_t depth, int numTrees) {
  cuda_check();
  size_t maxWidth = pow(2, depth);
  size_t numNode = maxWidth * 2 - 1;
  size_t numLeaves = numNode / 2 + 1;

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

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // get choice bits from receiver, hardcoded for now
  cudaMalloc(&d_otNodes, sizeof(*d_otNodes) * depth);

  // store tree in device memory
  TreeNode *d_Leaves;
  cudaMalloc(&d_Leaves, sizeof(*d_Leaves) * numLeaves);

  TreeNode *d_InputBuf;
  cudaMalloc(&d_InputBuf, sizeof(*d_InputBuf) * maxWidth / 2 + PADDED_LEN);

  // for storing the accumulated distributed-pprf
  TreeNode *d_sumLeaves;
  cudaMalloc(&d_sumLeaves, sizeof(*d_sumLeaves) * numLeaves);
  cudaMemset(d_sumLeaves, 0, sizeof(*d_sumLeaves) * numLeaves);

  for (int t = 0; t < numTrees; t++) {
    int puncturedIndex = 0;
    cudaMemcpy(d_Leaves, root, sizeof(*root), cudaMemcpyHostToDevice);
    size_t layerStartIdx = 1;
    while(otSent);

    for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_InputBuf, d_Leaves, sizeof(*d_Leaves) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_Leaves);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(texLKey, d_Leaves,  (unsigned*) d_InputBuf, 0, width);
      aesExpand128<<<grid, thread>>>(texRKey, d_Leaves,  (unsigned*) d_InputBuf, 1, width);

      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncturedIndex * 2 + 1 - (width - 1) + choice;
      cudaMemcpy(&d_otNodes[d-1], &d_Leaves[otLeafLayerIdx], sizeof(*d_Leaves), cudaMemcpyDeviceToDevice);
      puncturedIndex = puncturedIndex * 2 + 1 + (1 - choice);

      layerStartIdx += width;
    }

    accumulator<<<(numLeaves - 1) / 1024 + 1, 1024>>>(d_sumLeaves, d_Leaves, numLeaves);

    otSent = true;
    printf("Sender expansion %d complete\n", t);
  }
  cudaMemcpy(leaves, d_sumLeaves, sizeof(*leaves) * maxWidth, cudaMemcpyDeviceToHost);

  cudaFree(d_Leaves);
  cudaFree(d_InputBuf);
  cudaFree(d_sumLeaves);

  dealloc_key_texture(texLKey);
  dealloc_key_texture(texRKey);

  clock_gettime(CLOCK_MONOTONIC, &end);

  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESGPU sender: %0.4f ms\n", duration / NUM_SAMPLES);
}

void pprf_recver_gpu(TreeNode *leaves, size_t depth, int numTrees) {
  cuda_check();
  size_t maxWidth = pow(2, depth);
  size_t numNode = maxWidth * 2 - 1;
  size_t numLeaves = numNode / 2 + 1;

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

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // store tree in device memory
  TreeNode *d_Leaves;
  cudaMalloc(&d_Leaves, sizeof(*d_Leaves) * numLeaves);

  TreeNode *d_InputBuf;
  cudaMalloc(&d_InputBuf, sizeof(*d_InputBuf) * maxWidth / 2 + PADDED_LEN);

  // for storing the accumulated distributed-pprf
  TreeNode *d_multiPprf;
  cudaMalloc(&d_multiPprf, sizeof(*d_multiPprf) * numLeaves);
  cudaMemset(d_multiPprf, 0, sizeof(*d_multiPprf) * numLeaves);

  for(int t = 0; t < numTrees; t++) {
    while(!otSent);
    int choice = choices[t] & 1;
    int puncturedIndex = 2 - choice;

    cudaMemcpy(&d_Leaves[choice], &d_otNodes[0], sizeof(*d_otNodes), cudaMemcpyDeviceToDevice);

    size_t layerStartIdx = 3;
    for (size_t d = 2, width = 4; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      cudaMemcpy(d_InputBuf, d_Leaves, sizeof(*d_Leaves) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * sizeof(*d_Leaves);
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(texLKey, d_Leaves,  (unsigned*) d_InputBuf, 0, width);
      aesExpand128<<<grid, thread>>>(texRKey, d_Leaves,  (unsigned*) d_InputBuf, 1, width);

      cudaDeviceSynchronize();

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncturedIndex * 2 + 1 - (width - 1) + choice;
      cudaMemcpy(&d_Leaves[otLeafLayerIdx], &d_otNodes[d-1], sizeof(*d_Leaves), cudaMemcpyDeviceToDevice);
      puncturedIndex = puncturedIndex * 2 + 1 + (1 - choice);

      layerStartIdx += width;
    } 

    // todo: obtain y ^ delta from sender
    TreeNode puncturedNode;

    sum_pprf_into_sparse<<<(numLeaves - 1) / 1024 + 1, 1024>>>(d_multiPprf, d_Leaves, puncturedNode, numLeaves);
    cudaDeviceSynchronize();

    otSent = false;
    printf("Recver expansion %d complete\n", t);
  }

  cudaMemcpy(leaves, d_multiPprf, sizeof(*leaves) * maxWidth, cudaMemcpyDeviceToHost);

  cudaFree(d_Leaves);
  cudaFree(d_InputBuf);
  cudaFree(d_multiPprf);

  dealloc_key_texture(texLKey);
  dealloc_key_texture(texRKey);

  clock_gettime(CLOCK_MONOTONIC, &end);

  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESGPU receiver: %0.4f ms\n", duration / NUM_SAMPLES);
}
