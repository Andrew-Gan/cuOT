#ifndef __LOGIC_OPS_H__
#define __LOGIC_OPS_H__

#include "util.h"
#include "cufft.h"

__global__
void and_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void xor_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void poly_mod_gpu(uint64_t *data, uint64_t terms);

__global__
void and_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void xor_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void bit_transposer(uint64_t *out, uint64_t *in);

template<typename S, typename T>
__global__
void cast(S *input, T *output) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) printf("convert %f to %lu\n", (float)input[tid], (uint64_t)output[tid]);
  output[tid] = (T) input[tid];
}

__global__
void complex_dot_product(cufftComplex *c, cufftComplex *a, cufftComplex *b);

__global__
void xor_reduce_gpu(uint64_t *out, uint64_t *in);

__global__
void print_gpu(void *data, uint64_t n, uint64_t stride = 1);

#endif
