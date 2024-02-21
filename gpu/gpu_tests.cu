#include <cstdio>
#include <cuda.h>
#include <stdexcept>
#include <cassert>
#include "gpu_tests.h"
#include "gpu_ops.h"

int check_cuda() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	bool foundDev = false;
	int dev;
	printf("Found following devices:\n");
	for (dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (deviceProp.major >= 1) {
			printf("%d: %s\n", dev, deviceProp.name);
			foundDev = true;
		}
	}
	if (!foundDev)
		fprintf(stderr, "There is no device supporting CUDA.\n");
	return deviceCount;
}

void check_alloc(void *ptr) {
	int dev = 0;
	cudaGetDevice(&dev);
	printf("on device: %d\n", dev);
	cudaDeviceSynchronize();

	cudaPointerAttributes attr;
	uint64_t size = 0;

	if (cudaSuccess != cudaPointerGetAttributes(&attr, ptr))
		printf("Failed to get attribute\n");
	
	printf("ptr %p, dev %d, ", ptr, attr.device);

	if (dev != attr.device) {
		cudaSetDevice(attr.device);
	}

	if (CUDA_SUCCESS != cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr))
		printf("Failed to get range\n");

	printf("alloc %ld\n", size);
	fflush(stdout);
	cudaSetDevice(dev);
}

void check_call(const char* msg) {
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, msg);
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

bool check_rot(Mat &m0, Mat &m1, Mat &mc, uint64_t c) {
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

__global__
void _unpack_choice_bits(blk *out, uint64_t *choice, blk *delta) {
	uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if (choice[x / 64] >> (x % 64) & 1)
		out[x] = *delta;
}

bool check_cot(Mat &full, Mat &punc, Mat &choice, blk *delta) {
	Mat left({1, full.dim(0)});
	left.load(full.data());
	left ^= punc;
	Mat right({8*choice.size_bytes()});
	right.clear();
	uint64_t threads = choice.size() * BLOCK_BITS;
	uint64_t block = std::min(1024UL, threads);
	uint64_t grid = (threads + block - 1) / block;
	_unpack_choice_bits<<<grid, block>>>(right.data(), (uint64_t*) choice.data(), delta);
	return full == punc;
	// std::cout << "full\n" << full << std::endl;
	// std::cout << "punc\n" << punc << std::endl;
	// std::cout << "left\n" << left << std::endl;
	// std::cout << "right\n" << right << std::endl;
	return left == right;
}
