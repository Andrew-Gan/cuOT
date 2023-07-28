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

  printf("full\n");
  print_gpu<<<1, 1>>>((uint8_t*) fullVector.data(), 4, 16*512);
  cudaDeviceSynchronize();
  printf("punc\n");
  print_gpu<<<1, 1>>>((uint8_t*) puncVector.data(), 4, 16*512);
  cudaDeviceSynchronize();
  printf("choice\n");
  print_gpu<<<1, 1>>>((uint8_t*) choiceVector.data(), 4, 16*512);
  cudaDeviceSynchronize();
  printf("delta\n");
  print_gpu<<<1, 1>>>((uint8_t*) delta, 16);
  cudaDeviceSynchronize();

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
