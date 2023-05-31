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
void and_gpu(uint8_t *c, uint8_t *a, uint8_t *b, size_t n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n)
    c[x] = a[x] & b[x];
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

// sum_gpu<<<numElem / 2, elemSize>>>(c);
__global__
void sum_gpu(uint8_t *c) {
  size_t elemByte = threadIdx.x;
  size_t elemSize = blockDim.x;
  size_t leftId = blockIdx.x;
  for (size_t numElemRem = 2 * gridDim.x; numElemRem > 1; numElemRem /= 2) {
    size_t rightId = blockIdx.x + numElemRem / 2;
    if (rightId >= numElemRem) break;
    c[leftId * elemSize + elemByte] ^= c[rightId * elemSize + elemByte];
    __syncthreads();
  }
}
