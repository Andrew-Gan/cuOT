#include "util.h"

__global__
void xor_gpu(uint8_t *c, uint8_t *a, uint8_t *b, size_t n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] ^ b[x];
}

__global__
void xor_uneven(uint8_t *c, uint8_t *a, uint8_t *b, size_t len_b, size_t n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] ^ b[x % len_b];
}

__global__
void and_gpu(Vector c, Vector a, uint8_t b) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  c.data[x] = a.data[x] & b;
}

__global__
void print_gpu(uint8_t *a, size_t n) {
  for (int i = 0; i < n; i++)
    printf("%x ", a[i]);
  printf("\n");
}
