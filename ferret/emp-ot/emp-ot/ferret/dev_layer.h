#ifndef __CUDA_LAYER_H__
#define __CUDA_LAYER_H__

#include "gpu_tests.h"
#include "gpu_matrix.h"

#define NGPU 1

void cuda_mpcot_sender(Mat *expanded, blk *lSum_h, blk *rSum_h,
  blk *secret_sum, int tree, int depth, blk *Delta_f2k, int ngpu);

void cuda_mpcot_recver(Mat *expanded, blk *cSum_h, blk *secret_sum, int tree,
  int depth, bool *choices, int ngpu);

void cuda_gen_matrices(Role role, Mat *pubMats, int64_t n, int k, uint32_t *key, int ngpu);
void cuda_primal_lpn(Role role, Mat *pubMats, int64_t d, int64_t n, int k, Mat *expanded, blk *nn, blk *kk, int ngpu);

#endif
