#ifndef __GPU_TESTS_H__
#define __GPU_TESTS_H__

#include "gpu_matrix.h"

#define CHECK_CUDA(fn) {cudaError_t err = cudaDeviceSynchronize(); \
  if (err != cudaSuccess) { std::cerr << fn << ": " << cudaGetErrorString(err); }}

bool check_cuda(int minGPU);
void check_alloc(void *ptr);
bool check_rot(Mat &m0, Mat &m1, Mat &mc, uint64_t c);
bool check_cot(Mat &full, Mat &punc, Mat &choice, blk *delta);

#endif
