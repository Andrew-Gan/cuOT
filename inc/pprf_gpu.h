#ifndef __PPRF_GPU_H__
#define __PPRF_GPU_H__

#include "mytypes.h"

void pprf_sender_gpu(TreeNode *root, size_t depth, int numTree);
void pprf_recver_gpu(TreeNode *d_multiPprf, int *nonZeroRows, size_t depth, int numTree);

#endif
