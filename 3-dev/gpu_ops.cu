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

__device__
void clmul_128(uint64_t a, uint64_t b, uint64_t c[2]) {
  for (int i = 0; i < 64; i++) {
    if (a & (1 << i)) {
      c[0] ^= b << i;
      c[1] ^= b >> (64-i);
    }
  }
}

__device__
void shuffle_128(uint64_t a[2], uint8_t control, uint64_t b[2]) {
  uint32_t src[4] = {
    (uint32_t) a[0],
    (uint32_t) (a[0] >> 32),
    (uint32_t) a[1],
    (uint32_t) (a[1] >> 32),
  };
  uint32_t des[4];
  for (int i = 0; i < 4; i++) {
    uint8_t pos = (control >> (i * 2)) & 0b11;
    des[pos] = src[i];
  }
  b[0] = (uint64_t) des[1] << 32 | des[0];
  b[1] = (uint64_t) des[3] << 32 | des[2];
}

__global__
void complex_dot_product(cufftComplex *c_out, cufftComplex *a_in, cufftComplex *b_in) {
  uint64_t row = blockIdx.y;
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t width = 2 * gridDim.x * blockDim.x;
  uint64_t offset = row * width + 2 * tid;

  // KARATSUBA MULT
  uint64_t a[2] = { (uint64_t) (a_in[tid * 2].x), (uint64_t) (a_in[tid * 2 + 1].x) };
  uint64_t b[2] = { (uint64_t) (b_in[offset].x), (uint64_t) (b_in[offset + 1].x) };
  uint64_t c0[2];
  uint64_t c1[2];

  clmul_128(a[0], b[0], c0);
  clmul_128(a[1], b[1], c1);

  uint64_t tt0[2] = { a[0] ^ a[1], a[1] };
  uint64_t tt1[2] = { b[0] ^ b[1], b[1] };
  uint64_t tt2[2];
  clmul_128(tt0[0], tt1[0], tt2);
  tt2[0] ^= c0[0] ^ c1[0];
  tt2[1] ^= c0[1] ^ c1[1];

  c0[1] ^= tt2[0];
  c1[0] ^= tt2[1];

  // REDUCTION
  uint64_t reducer[2] = { 0x87, 0x0 };
  uint64_t x64[2];
  clmul_128(c1[1], reducer[0], x64);
  uint64_t out[2];
  shuffle_128(x64, 0xfe, out);
  c1[0] ^= out[0];
  c1[1] ^= out[1];
  shuffle_128(x64, 0x4f, out);
  c0[0] ^= out[0];
  c0[1] ^= out[1];
  clmul_128(c1[0], reducer[0], out);
  c0[0] ^= out[0];
  c0[1] ^= out[1];

  c_out[offset].x = (float) c0[0];
  c_out[offset].y = 0;
  c_out[offset + 1].x = (float) c0[1];
  c_out[offset + 1].y = 0;
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
void print_gpu(void *data, uint64_t n, uint64_t stride) {
  uint8_t *uData = (uint8_t*) data;
  for(int i = 0; i < n; i += 16) {
    for (int j = i; j < n && j < i + 16; j++)
      printf("%02x ", uData[j * stride]);
    printf("\n");
  }
}
