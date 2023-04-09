#ifndef __PPRF_GPU_H__
#define __PPRF_GPU_H__

#include <utility>
#include "mytypes.h"

std::pair<TreeNode*, uint64_t> pprf_sender_gpu(uint64_t *choices, TreeNode root, int depth, int numTrees);
std::pair<TreeNode*, int*> pprf_recver_gpu(uint64_t *choices, int depth, int numTrees);

#endif
