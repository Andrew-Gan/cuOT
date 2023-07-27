#include <assert.h>
#include <future>
#include "unit_test.h"
#include "expander.h"
#include "base_ot.h"

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
  assert(deviceCount > 0);
  assert(dev < deviceCount);
}

bool _cmp(OTblock &b0, OTblock &b1) {
  for (int i = 0; i < 4; i++) {
    if (b0.data[i] != b1.data[i])
      return false;
  }
  return true;
}

void test_base_ot() {
  const uint64_t choice = 0b1001;
  std::future sender = std::async([]() {
    return SimplestOT(Sender, 0, 4).send();
  });
  std::future recver = std::async([]() {
    return SimplestOT(Recver, 0, 4).recv(choice);
  });

  auto pair = sender.get();
  GPUdata m0_d = pair[0];
  GPUdata m1_d = pair[1];
  GPUdata mb_d = recver.get();

  OTblock m0[4], m1[4], mb[4];
  cudaMemcpy(m0, m0_d.data(), 4 * sizeof(OTblock), cudaMemcpyDeviceToHost);
  cudaMemcpy(m1, m1_d.data(), 4 * sizeof(OTblock), cudaMemcpyDeviceToHost);
  cudaMemcpy(mb, mb_d.data(), 4 * sizeof(OTblock), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; i++) {
    uint8_t c = choice & (1 << i);
    if (c == 0)
      assert(_cmp(mb[i], m0[i]));
    else
      assert(_cmp(mb[i], m1[i]));
  }

  printf("test_base_ot passed!\n");
}

void test_reduce() {
  GPUvector<OTblock> data(8);
  data.clear();
  OTblock buff;
  memset(&buff, 0, sizeof(OTblock));
  buff.data[0] = 0b1010;
  data.set(1, buff);
  buff.data[0] = 0b0101;
  data.set(2, buff);
  cudaStream_t s;
  cudaStreamCreate(&s);
  data.sum_async(1, 8, s);
  cudaDeviceSynchronize();
  cudaStreamDestroy(s);

  GPUvector<OTblock> data2(8);
  data.clear();
  buff.data[0] = 0b1110;
  data2.set(0, buff);

  assert(data == data2);
  printf("test_reduce passed!\n");
}

void test_cot(GPUvector<OTblock> &fullVector, OTblock *delta,
  GPUvector<OTblock> &puncVector, GPUvector<OTblock> &choiceVector) {

  // printf("full\n");
  // print_gpu<<<1, 1>>>((uint8_t*) fullVector.data(), 4, 16*512);
  // cudaDeviceSynchronize();
  // printf("punc\n");
  // print_gpu<<<1, 1>>>((uint8_t*) puncVector.data(), 4, 16*512);
  // cudaDeviceSynchronize();
  // printf("choice\n");
  // print_gpu<<<1, 1>>>((uint8_t*) choiceVector.data(), 4, 16*512);
  // cudaDeviceSynchronize();
  // printf("delta\n");
  // print_gpu<<<1, 1>>>((uint8_t*) delta, 16);
  // cudaDeviceSynchronize();

  fullVector ^= puncVector;
  choiceVector &= delta;

  // printf("lhs\n");
  // print_gpu<<<1, 1>>>((uint8_t*) fullVector.data(), 4, 16*512);
  // cudaDeviceSynchronize();
  // printf("rhs\n");
  // print_gpu<<<1, 1>>>((uint8_t*) choiceVector.data(), 4, 16*512);
  // cudaDeviceSynchronize();

  assert(fullVector == choiceVector);
}
