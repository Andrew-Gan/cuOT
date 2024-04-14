#ifndef __CUDA_LAYER_H__
#define __CUDA_LAYER_H__

#include "gpu_tests.h"
#include "gpu_matrix.h"

#define NGPU 1

void cuda_setdev(int gpu);
void cuda_malloc(void **ptr, size_t size);
void cuda_free(void *ptr);
void cuda_memcpy_H2D(void *des, void *src, size_t size);

void cuda_mpcot_sender(Mat *expanded, Mat *buffer, Mat *sep, blk *lSum_h,
  blk *rSum_h, blk *secret_sum, int tree, int depth, blk **delta, int ngpu);

void cuda_mpcot_recver(Mat *expanded, Mat *buffer, Mat *sep, blk *cSum_h,
  blk *secret_sum, int tree, int depth, bool *choices, int ngpu);

void cuda_primal_lpn(Role role, Mat *pubMats, int64_t d, int64_t n, int k,
  uint32_t *key, Mat *expanded, blk *nn, Mat *kk_d, blk *kk, int ngpu);

#endif
