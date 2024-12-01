#ifndef __CUDA_LAYER_H__
#define __CUDA_LAYER_H__

#include "gpu_tests.h"
#include "gpu_matrix.h"

// #define USE_CUSTOM_SPARSE_MAT_MULT

#define GPU_PARALLEL_FOR(fn) {						      \
  std::vector<future<void>> fut; 			          \
  for (int i = 0; i < ngpu; i++) {		    \
    fut.push_back(this->pool->enqueue([&, i](){ \
      cuda_setdev(i + party == ALICE ? 0 : ngpu);                   \
      fn 											                  \
    }));                                        \
  }                                             \
  for (auto & f : fut) f.get();                 \
}

void cuda_setdev(int gpu);

void cuda_ipc_open_mem_handle(void **ptr, uint8_t *handle);

void cuda_ipc_close_mem_handle(void *ptr);

void cuda_mpcot_sender(Mat &expanded, Mat &buffer, Mat &sep, blk *lSum_h,
  blk *rSum_h, blk *secret_sum, int t, int depth, blk *delta);

void cuda_mpcot_recver(Mat &expanded, Mat &buffer, Mat &sep, blk *cSum_h,
  blk *secret_sum, int t, int depth, bool *choices);

void cuda_primal_lpn(Mat &pubMats, int64_t d, int64_t n, int k,
  uint32_t *key, Mat &nn, const blk *kk);

void cuda_online_sender(const GPUdata &bo, const Mat &ch, Mat &data,
  int64_t length);
void cuda_online_recver(GPUdata &bo ,void *bo_other, GPUdata &b,
  const Mat &data, int64_t length);

#endif
