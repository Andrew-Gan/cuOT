#ifndef __RAND_GPU_H__
#define __RAND_GPU_H__

#include "mytypes.h"

Matrix gen_rand_gpu(size_t height, size_t width);
Matrix gen_ldpc_gpu(int numLeaves, int numTrees);

#endif
