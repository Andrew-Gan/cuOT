#include "basic_op.h"

__global__
void xor_gpu(uint8_t *c, uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] ^ b[x];
}

__global__
void xor_circular(uint8_t *c, uint8_t *a, uint8_t *b, uint64_t len_b, uint64_t n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] ^ b[x % len_b];
}

__global__
void and_gpu(uint8_t *c, uint8_t *a, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] &= a[x];
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__
void warpReduce(volatile uint64_t *sdata, uint64_t tid) {
  if (blockDim.x >= 64) sdata[tid] = sdata[tid] ^ sdata[tid + 32];
  if (blockDim.x >= 32) sdata[tid] = sdata[tid] ^ sdata[tid + 16];
  if (blockDim.x >= 16) sdata[tid] = sdata[tid] ^ sdata[tid + 8];
  if (blockDim.x >= 8) sdata[tid] = sdata[tid] ^ sdata[tid + 4];
  if (blockDim.x >= 4) sdata[tid] = sdata[tid] ^ sdata[tid + 2];
  // stop here for OTBlock reduction
  // if (blockDim.x >= 2) sdata[tid] ^= sdata[tid + 1];
}

__global__
void xor_reduce_gpu(uint64_t *g_data) {
  extern __shared__ uint64_t sdata[];
  uint64_t tid = threadIdx.x;
  uint64_t i = blockIdx.x * (blockDim.x * 2);

  sdata[tid] = g_data[tid + i] ^ g_data[tid + i + blockDim.x];
  if (blockDim.x >= 1024 && tid < 512) sdata[tid] ^= sdata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) sdata[tid] ^= sdata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) sdata[tid] ^= sdata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) sdata[tid] ^= sdata[tid + 64];
  __syncthreads();
  if (tid < 32) warpReduce(sdata, tid);
  if (tid < 2) g_data[2 * blockIdx.x + tid] = sdata[tid];
}
