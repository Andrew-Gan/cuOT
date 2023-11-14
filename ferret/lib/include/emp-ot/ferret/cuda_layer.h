#ifndef __CUDA_LAYER_H__
#define __CUDA_LAYER_H__

#include "gpu_vector.h"

typedef enum { H2H, H2D, D2H, D2D } cudaMemcpy_t;

void cuda_init();
void cuda_malloc(void **ptr, size_t n);
void cuda_memcpy(void *dest, void *src, size_t n, cudaMemcpy_t type);

void cuda_spcot_sender_compute(vec &tree, int t, int n, int depth, mat &lSum, mat &rSum);
void cuda_spcot_recver_compute(int t, int n, int depth, vec &tree, bool *b, mat &cSum);

void cuda_lpn_f2_compute(int d, int n, int k, uint32_t *key, vec &nn, blk *kk);

#endif
