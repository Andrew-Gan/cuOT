#include "unit_test.h"

__global__
void xor_gpu(Vector c, Vector a, Vector b) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  c.data[x] = a.data[x] ^ b.data[x];
}

__global__
void and_gpu(Vector c, Vector a, uint8_t b) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  c.data[x] = a.data[x] & b;
}

__global__
void cmp_gpu(bool *c, Vector a, Vector b) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  c[x] = a.data[x] == b.data[x];
}

// test A ^ C =  B & delta
//  delta should be 0b00000000 or 0b11111111
bool unit_test_correlation(Vector d_fullVec, Vector d_puncVec, Vector d_choiceVec, uint8_t delta) {
  int nBytes = d_fullVec.n / 8;
  
  Vector lhs = { .n = d_fullVec.n };
  cudaMalloc(&lhs.data, lhs.n / 8);
  xor_gpu<<<nBytes/ 1024, 1024>>>(lhs, d_fullVec, d_puncVec);

  Vector rhs = { .n = d_fullVec.n };
  cudaMalloc(&rhs.data, rhs.n / 8);
  and_gpu<<<nBytes / 1024, 1024>>>(rhs, d_choiceVec, delta);

  cudaDeviceSynchronize();

  bool *d_cmp, *cmp;
  cudaMalloc(&d_cmp, nBytes * sizeof(*d_cmp));

  cmp = new bool[nBytes];
  cudaMemcpy(cmp, d_cmp,  nBytes * sizeof(*d_cmp), cudaMemcpyDeviceToHost);
  
  int i = 0;
  while(i < nBytes) {
    if (cmp[i++] == false) {
      return false;
    }
  }

  return true;
}
