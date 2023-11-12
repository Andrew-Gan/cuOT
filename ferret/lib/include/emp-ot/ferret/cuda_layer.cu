#include "aes_op.h"
#include "expand.h"
#include "mat_mult.h"
#include "cuda_layer.h"

void cuda_init() {
	cudaFree(0);
}

void cuda_malloc(void **ptr, size_t n) {
	cudaMalloc(ptr, n);
}

void cuda_memcpy(void *dest, void *src, size_t n, cudaMemcpy_t type) {
	cudaMemcpy(dest, src, n, (cudaMemcpyKind)type);
}

void cuda_spcot_sender_compute(blk *tree, int n, int depth, vec &lSum, vec &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesHash aesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	vec sep(n);

	for (uint64_t d = 1, w = 2; d <= depth; d++, w *= 2) {
		aesHash.expand(tree, sep, tree, w); // implement inplace mode
		sep.sum(2, w / 2);
		cudaMemcpy(lSum.data(d-1), sep.data(0), sizeof(blk), cudaMemcpyDeviceToDevice);
		cudaMemcpy(rSum.data(d-1), sep.data(1), sizeof(blk), cudaMemcpyDeviceToDevice);
	}
}

void cuda_spcot_recver_compute(int n, int depth, blk *tree, bool *b, vec &cSum) {

	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesHash aesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	vec sep(n);
	uint64_t activeParent = 0;
	uint8_t choice;
	uint64_t offset;

	for (uint64_t d = 1, w = 2; d <= depth; d++, w *= 2) {
		aesHash.expand(tree, sep, tree, w); // implement inplace mode
		choice = b[d-1];
		offset = (w / 2) * choice + activeParent;
		cudaMemcpy(sep.data(offset), cSum.data(d-1), sizeof(blk), cudaMemcpyDeviceToDevice);
		if (d == depth) {
			offset = (w / 2) * (1-choice) + activeParent;
			cudaMemcpy(sep.data(offset), cSum.data(d), sizeof(blk), cudaMemcpyDeviceToDevice);
		}
		sep.sum(2, w / 2);
		offset = 2 * activeParent + choice;
		cudaMemcpy(tree + offset, sep.data(choice), sizeof(blk), cudaMemcpyDeviceToDevice);
		if (d == depth) {
			offset = 2 * activeParent + (1-choice);
			cudaMemcpy(tree + offset, sep.data(1-choice), sizeof(blk), cudaMemcpyDeviceToDevice);
		}

		activeParent *= 2;
		activeParent += 1 - choice;
	}
}

void cuda_lpn_f2_compute(int d, int n, int k, uint32_t *key, vec &nn, blk *kk) {
	blk *r_in, *r_out;
	cudaMalloc(&r_in, (d * n / 4) * sizeof(*r_in));
	cudaMalloc(&r_out, (d * n / 4) * sizeof(*r_out));

	dim3 grid(n/4/1024, d);
	make_block<<<grid, 1024>>>(r_in);
	aesEncrypt128<<<d*n/4/1024, 1024>>>((uint32_t*)key, (uint32_t*)r_out, (uint32_t*)r_in);
	lpn_single_row<<<n / 1024, 1024>>>((uint32_t*)r_out, d, k, nn.data(), kk);

	cudaDeviceSynchronize();
}
