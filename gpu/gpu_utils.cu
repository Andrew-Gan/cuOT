#include "gpu_utils.h"
#include <cstdio>
#include <cuda.h>
#include <stdexcept>
#include "gpu_vector.h"

void check_alloc(blk *ptr) {
	uint64_t size = 0;
	int dev = 0;
	cudaGetDevice(&dev);
	CUresult res = cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr);
	if (res != CUDA_SUCCESS) {
		printf("ptr %p, dev %d, alloc %ld\n", ptr, dev, size);
		throw std::runtime_error("something went wrong!\n");
	}
	fflush(stdout);
}

void check_call(const char* msg) {
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, msg);
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

bool check_rot(Vec &m0, Vec &m1, Vec &mc, uint8_t *c) {
	blk *b0 = new blk[1024], *b1 = new blk[1024], *bc = new blk[1024];

	for (int i = 0; i < mc.size(); i += 1024) {
		cudaMemcpy(b0, m0.data(i), 1024*sizeof(blk), cudaMemcpyDeviceToHost);
		cudaMemcpy(b1, m1.data(i), 1024*sizeof(blk), cudaMemcpyDeviceToHost);
		cudaMemcpy(bc, mc.data(i), 1024*sizeof(blk), cudaMemcpyDeviceToHost);

		for (int j = 0; j < 1024; j++) {
			uint8_t choiceBit = c[(i+j)/8] & (1<<(i+j%8));
			if (choiceBit == 0 && memcmp(&b0[i+j], &bc[i+j], sizeof(blk)) != 0
			 || choiceBit == 1 && memcmp(&b1[i+j], &bc[i+j], sizeof(blk)) != 0) {
				printf("Error at OT %d: failed to get correct message\n", i+j);
			}
		}
	}

	delete[] b0;
	delete[] b1;
	delete[] bc;
	printf("Passed check_rot\n");
}

bool check_cot(Vec &full, Vec &punc, uint64_t *choice, blk delta) {
	blk *bF = new blk[1024];
	blk *bP = new blk[1024];

	for (int i = 0; i < full.size(); i += 1024) {
		cudaMemcpy(bF, full.data(i), 1024*sizeof(blk), cudaMemcpyDeviceToHost);
		cudaMemcpy(bP, punc.data(i), 1024*sizeof(blk), cudaMemcpyDeviceToHost);

		for (int j = 0; j < 1024; j++) {
			uint8_t choiceBit = choice[(i+j)/64] & (1<<(i+j%64));
			blk deltaMsg;
			for (int k = 0; k < 4; k++)
				deltaMsg.data[k] = bF[i+j].data[k] ^ delta.data[k];
			if (choiceBit == 0 && memcmp(&bF[i+j], &bP[i+j], sizeof(blk)) != 0
			 || choiceBit == 1 && memcmp(&deltaMsg, &bP[i+j], sizeof(blk)) != 0) {
				printf("Error at OT %d: failed to get correct message\n", i+j);
			}
		}
	}

	delete[] bF;
	delete[] bP;
	printf("Passed check_rot\n");
}
