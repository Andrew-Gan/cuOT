#include "pprf_gpu.h"

std::atomic<TreeNode*>* d_otNodes = nullptr;
std::atomic<bool>* treeExpanded = nullptr;

__host__
void cuda_check() {
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
void xor_prf(TreeNode *sum, TreeNode *operand, size_t numLeaves) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numLeaves) {
    return;
  }
  for (int i = 0; i < TREENODE_SIZE / 4; i++) {
    sum[idx].data[i] ^= operand[idx].data[i];
  }
}
