#include "gpu_ops.h"
#include "gpu_define.h"

#define BIT_ACCESS(d, w, r, c) ((d[r * w + c / 64] >> (63-(c % 64))) & 0b1)

__global__
void gpu_and(uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) a[x] &= b[x];
}

__global__
void gpu_xor(uint8_t *a, uint8_t *b, uint64_t n, uint64_t rowBytes) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t y = blockIdx.y;
  if (x < n) a[y * rowBytes + x] ^= b[y * rowBytes + x];
}

__global__
void and_single(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) a[tid] &= b[tid % size];
}

__global__
void xor_single(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) a[tid] ^= b[tid % size];
}

__global__
void make_block(blk *blocks, uint64_t startIndex) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t *b64 = (uint64_t*)(&blocks[i]);
  b64[0] = 4 * ((i+startIndex) / 10);
  b64[1] = i % 10;
  blocks[i] = *(blk*)b64;
}

#ifndef USE_COALESCED_TREE_EXPANSION
// start with interleaved and end with separated
__global__
void separator(blk *out, blk *in) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < gridDim.x * blockDim.x / 2)
    out[i] = in[2*i];
  else
    out[i] = in[(2*(i-(gridDim.x * blockDim.x / 2))+1)];
}
#endif
