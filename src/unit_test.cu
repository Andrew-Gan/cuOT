#include <assert.h>
#include "unit_test.h"
#include "rsa.h"
#include "aes.h"

void test_rsa() {
  Rsa rsa;
  const char *input = "this is a test";
  uint8_t output[16] = {0};
  memcpy(output, input, 15);
  rsa.encrypt((uint32_t*) output, 15);
  assert(memcmp(input, output, 15) != 0);
  rsa.decrypt((uint32_t*) output, 15);
  assert(memcmp(input, output, 15) == 0);
}

void test_aes() {
  Aes aes;
  const char *sample = "this is a test";
  bool cmp[16];
  bool *d_cmp;
  cudaMalloc(&d_cmp, 16);

  AesBlocks input;
  cudaMemcpy(input.d_data, sample, 16, cudaMemcpyHostToDevice);
  AesBlocks output;
  cudaMemcpy(output.d_data, sample, 16, cudaMemcpyHostToDevice);
  aes.encrypt(&output);
  cmp_gpu<<<1, 16>>>(d_cmp, input.d_data, output.d_data);
  print_gpu<<<1, 1>>>(output.d_data, 16);
  cudaDeviceSynchronize();

  cudaMemcpy(cmp, d_cmp, 16, cudaMemcpyDeviceToHost);
  int i = 0, allEqual = true;
  while(i < 16) {
    if (!cmp[i]) {
      allEqual = false;
      break;
    }
  }
  assert(!allEqual);

  aes.decrypt(&output);
  print_gpu<<<1, 1>>>(output.d_data, 16);
  cmp_gpu<<<1, 16>>>(d_cmp, input.d_data, output.d_data);
  cudaDeviceSynchronize();

  cudaMemcpy(cmp, d_cmp, 16, cudaMemcpyDeviceToHost);
  int j = 0;
  allEqual = true;
  while(j < 16) {
    if (!cmp[j]) {
      allEqual = false;
      break;
    }
  }
  assert(allEqual);

  cudaFree(d_cmp);
}

// test A ^ C =  B & delta
//  delta should be 0b00000000 or 0b11111111
void test_cot(Vector d_fullVec, Vector d_puncVec, Vector d_choiceVec, uint8_t delta) {
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
  cmp_gpu<<<nBytes / 1024, 1024>>>(d_cmp, lhs.data, rhs.data);
  cudaDeviceSynchronize();

  cmp = new bool[nBytes];
  cudaMemcpy(cmp, d_cmp,  nBytes * sizeof(*d_cmp), cudaMemcpyDeviceToHost);

  cudaFree(lhs.data);
  cudaFree(rhs.data);
  cudaFree(d_cmp);

  int i = 0, allEqual = true;
  while(i < nBytes) {
    if (cmp[i++] == false) {
      allEqual = false;
    }
  }
  assert(allEqual);
}
