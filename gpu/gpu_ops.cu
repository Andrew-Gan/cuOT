#include <cstdio>
#include "gpu_ops.h"
#include "gpu_define.h"

#define BIT_ACCESS(d, w, r, c) ((d[r * w + c / 64] >> (63-(c % 64))) & 0b1)

__global__
void gpu_and(uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) a[x] &= b[x];
}

__global__
void gpu_xor(uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) a[x] ^= b[x];
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
void bit_transposer(uint8_t *out, uint8_t *in, dim3 grid) {
  uint64_t i = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
  uint64_t j = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t nRowBlocks = grid.y * blockDim.y;
  uint64_t bytesPerRow = grid.x * blockDim.x;

  uint64_t x =
    ( uint64_t( in[   i * 8       * bytesPerRow + j ] ) << 56 ) |
    ( uint64_t( in[ ( i * 8 + 1 ) * bytesPerRow + j ] ) << 48 ) |
    ( uint64_t( in[ ( i * 8 + 2 ) * bytesPerRow + j ] ) << 40 ) |
    ( uint64_t( in[ ( i * 8 + 3 ) * bytesPerRow + j ] ) << 32 ) |
    ( uint64_t( in[ ( i * 8 + 4 ) * bytesPerRow + j ] ) << 24 ) |
    ( uint64_t( in[ ( i * 8 + 5 ) * bytesPerRow + j ] ) << 16 ) |
    ( uint64_t( in[ ( i * 8 + 6 ) * bytesPerRow + j ] ) <<  8 ) |
    ( uint64_t( in[ ( i * 8 + 7 ) * bytesPerRow + j ] ) );
  uint64_t y =
    (x & 0x8040201008040201LL) |
    ((x & 0x0080402010080402LL) <<  7) |
    ((x & 0x0000804020100804LL) << 14) |
    ((x & 0x0000008040201008LL) << 21) |
    ((x & 0x0000000080402010LL) << 28) |
    ((x & 0x0000000000804020LL) << 35) |
    ((x & 0x0000000000008040LL) << 42) |
    ((x & 0x0000000000000080LL) << 49) |
    ((x >>  7) & 0x0080402010080402LL) |
    ((x >> 14) & 0x0000804020100804LL) |
    ((x >> 21) & 0x0000008040201008LL) |
    ((x >> 28) & 0x0000000080402010LL) |
    ((x >> 35) & 0x0000000000804020LL) |
    ((x >> 42) & 0x0000000000008040LL) |
    ((x >> 49) & 0x0000000000000080LL);
    out[ ( j * 8 ) * nRowBlocks + i ]     = uint8_t( ( y >> 56 ) & 0xFF );
    out[ ( j * 8 + 1 ) * nRowBlocks + i ] = uint8_t( ( y >> 48 ) & 0xFF );
    out[ ( j * 8 + 2 ) * nRowBlocks + i ] = uint8_t( ( y >> 40 ) & 0xFF );
    out[ ( j * 8 + 3 ) * nRowBlocks + i ] = uint8_t( ( y >> 32 ) & 0xFF );
    out[ ( j * 8 + 4 ) * nRowBlocks + i ] = uint8_t( ( y >> 24 ) & 0xFF );
    out[ ( j * 8 + 5 ) * nRowBlocks + i ] = uint8_t( ( y >> 16 ) & 0xFF );
    out[ ( j * 8 + 6 ) * nRowBlocks + i ] = uint8_t( ( y >> 8 ) & 0xFF );
    out[ ( j * 8 + 7 ) * nRowBlocks + i ] = uint8_t( y & 0xFF );
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__
void warp_reduce(uint64_t *sdata, uint64_t tid) {
  if (blockDim.x >= 64 && tid < 32) sdata[tid] ^= sdata[tid + 32];
  if (blockDim.x >= 32 && tid < 16) sdata[tid] ^= sdata[tid + 16];
  if (blockDim.x >= 16 && tid < 8) sdata[tid] ^= sdata[tid + 8];
  if (blockDim.x >= 8 && tid < 4) sdata[tid] ^= sdata[tid + 4];
  if (blockDim.x >= 4 && tid < 2) sdata[tid] ^= sdata[tid + 2];
}

__global__
void xor_reduce(uint64_t *out, uint64_t *in) {
  extern __shared__ uint64_t sdata[];
  uint64_t tid = threadIdx.x;
  uint64_t start = blockIdx.x * (2 * blockDim.x);

  sdata[tid] = in[start + tid] ^ in[start + tid + blockDim.x];
  __syncthreads();
  if (blockDim.x == 1024 && tid < 512) sdata[tid] ^= sdata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) sdata[tid] ^= sdata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) sdata[tid] ^= sdata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) sdata[tid] ^= sdata[tid + 64];
  __syncthreads();
  if (tid < 32) warp_reduce(sdata, tid);
  if (tid < 2) out[2 * blockIdx.x + tid] = sdata[tid];
}

__global__
void print(uint8_t *data, uint64_t n, uint64_t stride) {
  for(int i = 0; i < n; i += 16) {
    for (int j = i; j < n && j < i + 16; j++)
      printf("%02x ", data[j * stride]);
    printf("\n");
  }
}

__global__
void print(float *data, uint64_t n, uint64_t stride) {
  for(int i = 0; i < n; i += 32) {
    for (int j = i; j < n && j < i + 32; j++)
      printf("%.0f ", data[j * stride]);
    printf("\n");
  }
}

__global__
void print(cuComplex *data, uint64_t n, uint64_t stride) {
  for(int i = 0; i < n; i += 16) {
    for (int j = i; j < n && j < i + 16; j++)
      printf("%.2f + %.2fi ", data[j * stride].x, data[j * stride].y);
    printf("\n");
  }
}
