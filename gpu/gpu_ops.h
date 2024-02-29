#ifndef __LOGIC_OPS_H__
#define __LOGIC_OPS_H__

#include "cufft.h"
#include <cstdint>
#include <type_traits>
#include "gpu_define.h"

__global__
void gpu_and(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void gpu_xor(uint8_t *a, uint8_t *b, uint64_t n, uint64_t rowBytes = 0);

__global__
void and_single(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void xor_single(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void bit_transposer(uint8_t *out, uint8_t *in);

__global__
void xor_reduce(uint64_t *out, uint64_t *in);

__global__
void print(uint8_t *data, uint64_t n, uint64_t stride = 1);

__global__
void print(float *data, uint64_t n, uint64_t stride = 1);

__global__
void print(cuComplex *data, uint64_t n, uint64_t stride = 1);

cudaError_t cudaMemcpy2DPeerAsync(void *dst, size_t dpitch, int dstDevice,
  const void *src, size_t spitch, int srcDevice, size_t width, size_t height,
  cudaStream_t *s);

#endif
