#ifndef __DEV_LAYER_H__
#define __DEV_LAYER_H__

#include "gpu_define.h"
#include "gpu_span.h"

void LpnF2_LpnF2_dev(uint32_t *rdKey, Mat &pubMat);
void SPCOT_recver_compute_dev(uint64_t tree_n, Mat &cSum, uint64_t inWidth,
  uint64_t *activeParent, Mat &separated, Span &tree, uint64_t depth,
  uint64_t d, bool *choice);
void LpnF2_encode_dev(Mat &pubMat, uint64_t n, uint64_t k, uint64_t d, Span &nn, Span &kk);
void set_dev(int dev);
void malloc_dev(void **mem, size_t size);
void free_dev(void *mem);
void memset_dev(void *des, int val, size_t size);
void memcpy_H2D_dev(void *des, void *src, size_t size);
void memcpy_D2H_dev(void *des, void *src, size_t size);
void memcpy_D2D_dev(void *des, void *src, size_t size);
void sync_dev();

#endif