#include "gpu_tools.h"
#include <cstdio>
#include <cuda.h>

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