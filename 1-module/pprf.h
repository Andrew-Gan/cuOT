#ifndef __PPRF_GPU_H__
#define __PPRF_GPU_H__

#include <utility>
#include <atomic>
#include <array>

#include "util.h"
#include "gpu_block.h"

std::pair<GPUBlock, GPUBlock> pprf_sender(TreeNode root, int depth, int numTrees);
std::pair<GPUBlock, SparseVector> pprf_recver(uint64_t *choices, int depth, int numTrees);

#endif
