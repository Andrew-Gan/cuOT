#include <assert.h>
#include <future>
#include "unit_test.h"
#include "aes.h"
#include "simplest_ot.h"
#include "basic_op.h"

void test_cuda() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    fprintf(stderr, "There is no device.\n");
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major >= 1)
      break;
  }
  if (dev == deviceCount)
    fprintf(stderr, "There is no device supporting CUDA.\n");
  else
    cudaSetDevice(dev);
  printf("test_cuda passed!\n");
}

// void test_rsa() {
//   Rsa rsa;
//   const char *sample = "this is a test";
//   GPUBlock input(16);
//   GPUBlock output(16);
//   input.set((uint8_t*) sample, 15);
//   output.set((uint8_t*) sample, 15);
//   rsa.encrypt(output);
//   assert(input != output);
//   rsa.decrypt(output);
//   assert(input == output);
//   printf("test_rsa passed!\n");
// }

void test_aes() {
  Aes aes0;
  const char *sample = "this is a test";
  bool cmp[16];
  bool *cmp_d;
  cudaMalloc(&cmp_d, 16);

  GPUBlock input(16), buffer(16);
  input.set((const uint8_t*) sample, 16);
  buffer.set((const uint8_t*)sample, 16);

  aes0.encrypt(buffer);

  Aes aes1(aes0.key);
  aes1.decrypt(buffer);
  // cmp_gpu<<<1, 16>>>(cmp_d, input.data_d, buffer.data_d);
  // cudaDeviceSynchronize();

  cudaMemcpy(cmp, cmp_d, 16, cudaMemcpyDeviceToHost);
  int j = 0;
  bool allEqual = true;
  while(j < 16) {
    if (!cmp[j++]) {
      allEqual = false;
      break;
    }
  }
  assert(allEqual);
  cudaFree(cmp_d);
  printf("test_aes passed!\n");
}

void senderFunc(GPUBlock &m0, GPUBlock &m1) {
  SimplestOT sender(Sender, 0);
  sender.send(m0, m1);
}

GPUBlock recverFunc(uint8_t b) {
  SimplestOT recver(Recver, 0);
  GPUBlock mb = recver.recv(b);
  return mb;
}

void test_base_ot() {
  GPUBlock m0, m1, mb;
  m0.set(32);
  m1.set(64);

  std::future sender = std::async(senderFunc, std::ref(m0), std::ref(m1));
  std::future recver = std::async(recverFunc, 0);
  sender.get();
  mb = recver.get();
  assert(mb == m0);

  sender = std::async(senderFunc, std::ref(m0), std::ref(m1));
  recver = std::async(recverFunc, 1);
  sender.get();
  mb = recver.get();
  assert(mb == m1);

  printf("test_base_ot passed!\n");
}

// test A ^ C =  B & delta
//  delta should be 0b00000000 or 0b11111111
void test_cot(Vector fullVec_d, Vector puncVec_d, Vector choiceVec_d, uint8_t delta) {
  int nBytes = fullVec_d.n / 8;

  Vector lhs = { .n = fullVec_d.n };
  cudaMalloc(&lhs.data, lhs.n / 8);
  xor_gpu<<<nBytes/ 1024, 1024>>>(lhs.data, fullVec_d.data, puncVec_d.data, lhs.n);

  Vector rhs = { .n = fullVec_d.n };
  cudaMalloc(&rhs.data, rhs.n / 8);
  and_gpu<<<nBytes / 1024, 1024>>>(rhs, choiceVec_d, delta);

  cudaDeviceSynchronize();

  bool *cmp_d, *cmp;
  cudaMalloc(&cmp_d, nBytes * sizeof(*cmp_d));
  // cmp_gpu<<<nBytes / 1024, 1024>>>(cmp_d, lhs.data, rhs.data);
  // cudaDeviceSynchronize();

  cmp = new bool[nBytes];
  cudaMemcpy(cmp, cmp_d,  nBytes * sizeof(*cmp_d), cudaMemcpyDeviceToHost);

  cudaFree(lhs.data);
  cudaFree(rhs.data);
  cudaFree(cmp_d);

  int i = 0, allEqual = true;
  while(i < nBytes) {
    if (cmp[i++] == false) {
      allEqual = false;
    }
  }
  delete[] cmp;
  assert(allEqual);
  printf("test_cot passed!\n");
}
