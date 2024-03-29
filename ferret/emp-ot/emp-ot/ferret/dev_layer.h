#ifndef __CUDA_LAYER_H__
#define __CUDA_LAYER_H__

#include "gpu_tests.h"
#include "gpu_matrix.h"
#include "gpu_span.h"

typedef enum { H2H, H2D, D2H, D2D } cudaMemcpy_t;

void cuda_init(int party);
void cuda_malloc(void **ptr, size_t n);
void cuda_memcpy(void *dest, void *src, size_t n, cudaMemcpy_t type);
void cuda_free(void *ptr);

void cuda_spcot_sender_compute(Span *tree, int t, int depth, Mat &lSum, Mat &rSum);
void cuda_spcot_recver_compute(Span *tree, int t, int depth, Mat &cSum, bool *b);

void cuda_gen_matrices(Mat &pubMat, uint32_t *key);
void cuda_lpn_f2_compute(blk *pubMat, int d, int n, int k, Span &nn, Span &kk);

#endif