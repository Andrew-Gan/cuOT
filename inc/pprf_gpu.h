#ifndef __PPRF_GPU_H__
#define __PPRF_GPU_H__

#include <utility>
#include <atomic>
#include <array>

#include "util.h"

extern std::atomic<TreeNode*>* d_otNodes;
extern std::atomic<bool>* treeExpanded;

std::pair<Vector, uint64_t> pprf_sender_gpu(uint64_t *choices, TreeNode root, int depth, int numTrees);
std::pair<Vector, Vector> pprf_recver_gpu(uint64_t *choices, int depth, int numTrees);

__host__
void cuda_check();

__global__
void xor_prf(TreeNode *sum, TreeNode *operand, size_t numLeaves);

#endif
