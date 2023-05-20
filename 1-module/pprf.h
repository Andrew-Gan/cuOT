#ifndef __PPRF_GPU_H__
#define __PPRF_GPU_H__

#include <utility>
#include <atomic>
#include <array>

#include "util.h"
#include "gpu_block.h"

std::pair<GPUBlock, uint64_t> pprf_sender(TreeNode root, int depth, int numTrees);
std::pair<Vector, SparseVector> pprf_recver(uint64_t *choices, int depth, int numTrees);

__global__
void xor_prf(TreeNode *sum, TreeNode *operand, size_t numLeaves);

#endif
