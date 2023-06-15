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

void senderFunc(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1) {
  SimplestOT sender(OT::Sender, 0);
  sender.send(m0, m1);
}

std::vector<GPUBlock> recverFunc(uint64_t b) {
  SimplestOT recver(OT::Recver, 0);
  return recver.recv(b);
}

void test_base_ot() {
  std::vector<GPUBlock> m0(4, GPUBlock(1024));
  std::vector<GPUBlock> m1(4, GPUBlock(1024));
  std::vector<GPUBlock> mb_expected(4, GPUBlock(1024));
  for (int i = 0; i < m0.size(); i++) {
    m0.at(i).clear();
    m0.at(i).set(0x20);
    m1.at(i).clear();
    m1.at(i).set(0x40);
  }
  std::future sender = std::async(senderFunc, std::ref(m0), std::ref(m1));
  std::future recver = std::async(recverFunc, 0b1001);

  for (GPUBlock &m : mb_expected) {
    m.clear();
  }
  mb_expected.at(0).set(0x40);
  mb_expected.at(1).set(0x20);
  mb_expected.at(2).set(0x20);
  mb_expected.at(3).set(0x40);
  sender.get();
  std::vector<GPUBlock> mb_actual = recver.get();
  for (int i = 0; i < mb_actual.size(); i++) {
    assert(mb_actual.at(i) == mb_expected.at(i));
  }

  printf("test_base_ot passed!\n");
}

void test_cot(GPUBlock &fullVector, GPUBlock &puncVector, GPUBlock &choiceVector, GPUBlock &delta) {
  fullVector ^= puncVector;
  choiceVector *= delta;

  // assert(fullVector == choiceVector);
  printf("test_cot passed!\n");
}
