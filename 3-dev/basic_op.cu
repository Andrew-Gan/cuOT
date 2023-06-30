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

__device__
void modular_exp_helper(uint32_t *b, uint32_t e, uint32_t n) {
  uint64_t c = 1;
  for (int i = 0; i < e; i++) {
    c = (uint64_t)(*b * c) % n;
  }
  *b = c;
}

__global__
void modular_exp_gpu(uint32_t *b, uint32_t e, uint32_t n) {
  modular_exp_helper(b, e, n);
}

__global__
void chinese_rem_theorem_gpu(uint32_t *c, uint32_t d, uint32_t p,
  uint32_t q, uint32_t d_p, uint32_t d_q, uint32_t q_inv) {

  uint32_t m1 = *c;
  uint32_t m2 = *c;
  modular_exp_helper(&m1, d_p, p);
  modular_exp_helper(&m2, d_q, q);
  uint64_t h = ((uint64_t) q_inv * (m1 - m2)) % p;
  *c = (m2 + h * q) % (p * q);
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__
void warpReduce(volatile uint32_t *sdata, uint64_t tid) {
  if (blockDim.x >= 64) sdata[tid] = sdata[tid] ^ sdata[tid + 32];
  if (blockDim.x >= 32) sdata[tid] = sdata[tid] ^ sdata[tid + 16];
  if (blockDim.x >= 16) sdata[tid] = sdata[tid] ^ sdata[tid + 8];
  if (blockDim.x >= 8) sdata[tid] = sdata[tid] ^ sdata[tid + 4];
  // stop here for OTBlock reduction
  // if (blockDim.x >= 4) sdata[tid] ^= sdata[tid + 2];
  // if (blockDim.x >= 2) sdata[tid] ^= sdata[tid + 1];
}

__global__
void xor_reduce_gpu(uint32_t *g_data, uint64_t n) {
  extern __shared__ uint32_t sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + tid;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { sdata[tid] ^= g_data[i] + g_data[i+blockDim.x]; i ^= gridSize; }
  __syncthreads();
  if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] ^= sdata[tid + 256]; } __syncthreads(); }
  if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] ^= sdata[tid + 128]; } __syncthreads(); }
  if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] ^= sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce(sdata, tid);
  if (tid < (sizeof(OTBlock) / 4)) g_data[blockIdx.x + tid] = sdata[tid];
}
