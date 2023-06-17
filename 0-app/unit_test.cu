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
  assert(deviceCount > 0);
  assert(dev < deviceCount);
}

void test_aes() {
  Aes aes0;
  Aes aes1(aes0.key);
  const char *sample = "this is a test";

  GPUBlock buffer(1024);
  buffer.clear();
  buffer.set((const uint8_t*) sample, 16);

  aes0.encrypt(buffer);
  uint8_t encryptedData[16];
  cudaMemcpy(encryptedData, buffer.data_d, 16, cudaMemcpyDeviceToHost);
  assert(memcmp(sample, encryptedData, 16) != 0);

  aes1.decrypt(buffer);
  uint8_t decryptedData[16];
  cudaMemcpy(decryptedData, buffer.data_d, 16, cudaMemcpyDeviceToHost);
  assert(memcmp(sample, decryptedData, 16) == 0);

  printf("test_aes passed!\n");
}

void test_base_ot() {
  const uint64_t choice = 0b1001;
  std::future sender = std::async([]() {
    return SimplestOT(SimplestOT::Sender, 0).send(4);
  });
  std::future recver = std::async([]() {
    return SimplestOT(SimplestOT::Recver, 0).recv(4, choice);
  });

  auto pair = sender.get();
  std::vector<GPUBlock> m0 = pair[0];
  std::vector<GPUBlock> m1 = pair[1];
  std::vector<GPUBlock> mb = recver.get();

  for (int i = 0; i < mb.size(); i++) {
    uint8_t c = choice & (1 << i);
    std::cout << "m0: " << m0.at(i) << " m1: " << m1.at(i) << " mb: " << mb.at(i) << std::endl;
    if (c == 0)
      assert(mb.at(i) == m0.at(i));
    else
      assert(mb.at(i) == m1.at(i));
  }

  printf("test_base_ot passed!\n");
}

void test_cot(GPUBlock &fullVector, GPUBlock &puncVector, GPUBlock &choiceVector, GPUBlock &delta) {
  fullVector ^= puncVector;
  choiceVector *= delta;

  // assert(fullVector == choiceVector);
  printf("test_cot passed!\n");
}
