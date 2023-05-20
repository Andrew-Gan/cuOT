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
  GPUBlock m0(1024), m1(1024), mb(1024);
  m0.set(0x20);
  m1.set(0x40);
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

void test_cot(Vector fullVec_d, Vector puncVec_d, Vector choiceVec_d, uint8_t delta) {
  GPUBlock fullVector(fullVec_d.nBits / 8);
  GPUBlock puncVector(puncVec_d.nBits / 8);
  GPUBlock choiceVector(choiceVec_d.nBits / 8);

  fullVector.set(fullVec_d.data, fullVec_d.nBits / 8);
  puncVector.set(puncVec_d.data, puncVec_d.nBits / 8);
  choiceVector.set(choiceVec_d.data, choiceVec_d.nBits / 8);

  GPUBlock lhs = fullVector ^ puncVector;
  GPUBlock rhs = choiceVector * delta;

  assert(lhs == rhs);
  printf("test_cot passed!\n");
}
