#ifndef __LOGIC_OPS_H__
#define __LOGIC_OPS_H__

#include "util.h"
#include "cufft.h"

#include <type_traits>

__global__
void and_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void xor_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void and_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void xor_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void bit_transposer(uint8_t *out, uint8_t *in, dim3 grid);

template<typename S, typename T>
__global__
void cast(S *input, T *output) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  output[tid] = (T) (input[tid]);
  // if (tid == 0 && std::is_floating_point<S>::value && std::is_integral<T>::value)
    // printf("f to d convert %f to %lu\n", (float)(input[tid]), (uint64_t)(output[tid]));
  // if (tid == 0 && std::is_integral<S>::value && std::is_floating_point<T>::value)
    // printf("d to f convert %lu to %f\n", (uint64_t)(input[tid]), (float)(output[tid]));
}

__global__
void complex_dot_product(cufftComplex *c, cufftComplex *a, cufftComplex *b);

__global__
void xor_reduce_gpu(uint64_t *out, uint64_t *in);

__global__
void print_gpu(void *data, uint64_t n, uint64_t stride = 1);

#endif
