#include <assert.h>
#include <future>
#include "unit_test.h"
#include "expand.h"
#include "base_ot.h"

void test_cuda() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  assert(deviceCount >= 2);

  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major >= 1)
      break;
  }
  if (dev == deviceCount)
    fprintf(stderr, "There is no device supporting CUDA.\n");
  assert(dev < deviceCount);
}

bool _cmp(blk &b0, blk &b1) {
  for (int i = 0; i < 4; i++) {
    if (b0.data[i] != b1.data[i])
      return false;
  }
  return true;
}

void test_reduce() {
  GPUvector<blk> data(8);
  data.clear();
  blk buff;
  memset(&buff, 0, sizeof(blk));
  buff.data[0] = 0b1010;
  data.set(1, buff);
  buff.data[0] = 0b0101;
  data.set(2, buff);
  cudaStream_t s;
  cudaStreamCreate(&s);
  data.sum_async(1, 8, s);
  cudaDeviceSynchronize();
  cudaStreamDestroy(s);

  GPUvector<blk> data2(8);
  data.clear();
  buff.data[0] = 0b1110;
  data2.set(0, buff);

  assert(data == data2);
  printf("test_reduce passed!\n");
}

void test_cot(SilentOTSender &sender, SilentOTRecver &recver) {
  GPUvector<blk> lhs(recver.puncVector.size());
  cudaMemcpyPeer(lhs.data(), 0, recver.puncVector.data(), 1, recver.puncVector.size_bytes());

  GPUvector<blk> rhs(recver.choiceVector.size());
  cudaMemcpyPeer(rhs.data(), 0, recver.choiceVector.data(), 1, recver.choiceVector.size_bytes());

  printf("fullVector\n");
  print<<<1, 1>>>(sender.fullVector.data(), 64, 16);
  cudaDeviceSynchronize();

  printf("puncVector\n");
  print<<<1, 1>>>(lhs.data(), 64, 16);
  cudaDeviceSynchronize();

  printf("choiceVector\n");
  print<<<1, 1>>>(rhs.data(), 64, 16);
  cudaDeviceSynchronize();

  lhs ^= sender.fullVector;
  rhs &= sender.delta;

  assert(lhs == rhs);

  printf("correlation test passed!\n");
}
