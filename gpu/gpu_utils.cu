#include "gpu_utils.h"
#include <cstdio>
#include <cuda.h>
#include <stdexcept>

void check_alloc(blk *ptr) {
	uint64_t size = 0;
	int dev = 0;
	cudaGetDevice(&dev);
	CUresult res = cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr);
	printf("ptr %p, dev %d, alloc %ld\n", ptr, dev, size);
	if (res != CUDA_SUCCESS)
		printf("something went wrong!\n");
	fflush(stdout);
}

void check_call(const char* msg) {
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, msg);
		throw std::runtime_error(cudaGetErrorString(err));
	}
}
