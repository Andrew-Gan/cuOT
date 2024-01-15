#include <cstdio>
#include <cuda.h>
#include <stdexcept>
#include <cassert>
#include "gpu_tests.h"
#include "gpu_ops.h"

// comment out during benchmarking
// #define CHECK_ALLOC
// #define CHECK_CALL

void check_cuda() {
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

void check_alloc(void *ptr) {
#ifdef CHECK_ALLOC
	int dev = 0;
	cudaGetDevice(&dev);
	printf("currently on device: %d\n", dev);
	cudaDeviceSynchronize();

	cudaPointerAttributes attr;
	uint64_t size = 0;

	if (cudaSuccess != cudaPointerGetAttributes(&attr, ptr))
		printf("Failed to get attribute\n");

	if (CUDA_SUCCESS != cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr))
		printf("Failed to get range\n");

	printf("ptr %p, dev %d, alloc %ld\n", ptr, attr.device, size);
	fflush(stdout);
#endif
}

void check_call(const char* msg) {
#ifdef CHECK_CALL
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, msg);
		throw std::runtime_error(cudaGetErrorString(err));
	}
#endif
}

void check_mem() {
	cudaDeviceSynchronize();

	int dev = 0;
	cudaGetDevice(&dev);
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("dev %d: CUDA free memory: ", dev);
	if ((free >> 30) > 0)
		printf("%lu / %lu GB\n", free >> 30, total >> 30);
	else if ((free >> 20) > 0)
		printf("%lu / %lu MB\n", free >> 20, total >> 20);
	else if ((free >> 10) > 0)
		printf("%lu / %lu KB\n", free >> 10, total >> 10);
	else
		printf("%lu / %lu B\n", free, total);
}

bool check_rot(Vec &m0, Vec &m1, Vec &mc, uint64_t c) {
	int numTree = mc.size();
	blk *b0 = new blk[numTree], *b1 = new blk[numTree], *bc = new blk[numTree];

	cudaMemcpy(b0, m0.data(), m0.size_bytes(), cudaMemcpyDeviceToHost);
	cudaMemcpy(b1, m1.data(), m1.size_bytes(), cudaMemcpyDeviceToHost);
	cudaMemcpy(bc, mc.data(), mc.size_bytes(), cudaMemcpyDeviceToHost);

	for (int t = 0; t < numTree; t++) {
		uint8_t choiceBit = c & 1;
		if (choiceBit == 0 && memcmp(&b0[t], &bc[t], sizeof(blk)) != 0
		 || choiceBit == 1 && memcmp(&b1[t], &bc[t], sizeof(blk)) != 0) {
			printf("Error at ROT %d\n", t);
			return false;
		}
		c >>= 1;
	}

	delete[] b0;
	delete[] b1;
	delete[] bc;
	return true;
}

bool check_cot(Vec &full, Vec &punc, Vec &choice, blk *delta) {
	blk *delta_d;
	cudaMalloc(&delta_d, sizeof(*delta_d));
	cudaMemcpy(delta_d, delta, sizeof(*delta_d), cudaMemcpyHostToDevice);

	Vec left(punc);
	left ^= full;
	Vec right(choice);
	right &= delta_d;

	cudaFree(delta_d);
	return left == right;
}
