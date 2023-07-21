#include "gpu_ops.h"

#define BIT_ACCESS(d, w, r, c) ((d[r * w + c / 64] >> (63-(c % 64))) & 0b1)

__global__
void and_gpu(uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) a[x] &= b[x];
}

__global__
void xor_gpu(uint8_t *a, uint8_t *b, uint64_t n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) a[x] ^= b[x];
}

__global__
void poly_mod_gpu(uint64_t *data, uint64_t terms) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t reducedTerms = gridDim.x * blockDim.x;
  for (uint64_t i = 1; i < terms / reducedTerms; i++) {
    data[tid] += data[i * reducedTerms + tid];
  }
}

__global__
void and_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) a[tid] &= b[tid % size];
}

__global__
void xor_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) a[tid] ^= b[tid % size];
}

__global__
void bit_transposer(uint64_t *out, uint64_t *in) {
  // matrix dimensions
  uint64_t colsU64In = gridDim.x * blockDim.x / 64;
  uint64_t colsU64Out = gridDim.y * blockDim.y;

  uint64_t rowOut = (blockIdx.x * blockDim.x + threadIdx.x);
  uint64_t colOut = (blockIdx.y * blockDim.y + threadIdx.y);
  uint64_t colIn = rowOut;
  uint64_t res = 0;

  for (uint8_t i = 0; i < 64; i++) {
    uint64_t rowIn = 8 * colOut + i;
    res |= BIT_ACCESS(in, colsU64In, rowIn, colIn) << (63-i);
  }
  out[rowOut * colsU64Out + colOut] = res;
}

__global__
void int_to_float(float *o, uint64_t *i) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  o[tid] = (float) i[tid];
}

__global__
void float_to_int(uint64_t *o, float *i) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  o[tid] = (uint64_t) i[tid];
}

__global__
void complex_dot_product(cufftComplex *c, cufftComplex *a, cufftComplex *b) {
  uint64_t row = blockIdx.y;
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t width = gridDim.x * blockDim.x;
  uint64_t offset = row * width + tid;
  c[offset].x = a[tid].x * b[offset].x + a[tid].y * b[offset].y;
  c[offset].y = a[tid].x * b[offset].y + a[tid].y * b[offset].x;
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__
void warp_reduce(volatile uint64_t *sdata, uint64_t tid) {
  if (blockDim.x >= 64) sdata[tid] = sdata[tid] ^ sdata[tid + 32];
  if (blockDim.x >= 32) sdata[tid] = sdata[tid] ^ sdata[tid + 16];
  if (blockDim.x >= 16) sdata[tid] = sdata[tid] ^ sdata[tid + 8];
  if (blockDim.x >= 8) sdata[tid] = sdata[tid] ^ sdata[tid + 4];
  if (blockDim.x >= 4) sdata[tid] = sdata[tid] ^ sdata[tid + 2];
  // stop here for OTblock reduction
  // if (blockDim.x >= 2) sdata[tid] ^= sdata[tid + 1];
}

__global__
void xor_reduce_gpu(uint64_t *data) {
  extern __shared__ uint64_t sdata[];
  uint64_t tid = threadIdx.x;
  uint64_t start = blockIdx.x * (blockDim.x * 2);

  sdata[tid] = data[start + tid] ^ data[start + tid + blockDim.x];
  __syncthreads();
  if (blockDim.x >= 1024 && tid < 512) sdata[tid] ^= sdata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) sdata[tid] ^= sdata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) sdata[tid] ^= sdata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) sdata[tid] ^= sdata[tid + 64];
  __syncthreads();
  if (tid < 32) warp_reduce(sdata, tid);
  if (tid < 2) data[start + tid] = sdata[tid];
}

__global__
void xor_reduce_packer_gpu(uint64_t *data, uint64_t width) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  data[2 * tid] = data[tid * width];
  data[2 * tid + 1] = data[tid * width + 1];
}

__global__
void print_gpu(uint8_t *data, uint64_t n) {
  for(int i = 0; i < n; i+= 16) {
    for (int j = i; j < n && j < i + 16; j++)
      printf("%x ",  data[j]);
    printf("\n");
  }
}
