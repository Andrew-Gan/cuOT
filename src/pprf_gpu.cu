#include "pprf_gpu.h"

std::atomic<TreeNode*>* d_otNodes = nullptr;
std::atomic<bool>* treeExpanded = nullptr;

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
