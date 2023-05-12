#include "basic_op.h"

__global__
void xor_gpu(uint8_t *c, uint8_t *a, uint8_t *b, size_t n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] ^ b[x];
}

__global__
void xor_circular(uint8_t *c, uint8_t *a, uint8_t *b, size_t len_b, size_t n) {
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
