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

void cuda_spcot_sender_compute(vec &tree, int t, int n, int depth, mat &lSum, mat &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesHash aesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	vec separated(t*n);
	for (uint64_t d = 0, w = 1; d < depth; d++, w *= 2) {
		aesHash.expand(tree, separated, tree, w*t); // implement inplace mode
		separated.sum(2*t, w);
		cudaMemcpy(lSum.data(d, 0), separated.data(0), t*sizeof(blk), cudaMemcpyDeviceToDevice);
		cudaMemcpy(rSum.data(d, 0), separated.data(t), t*sizeof(blk), cudaMemcpyDeviceToDevice);
	}

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		printf("spcot_sender: %s\n", cudaGetErrorString(err));
}

void cuda_spcot_recver_compute(int t, int n, int depth, vec &tree, bool *b, mat &cSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesHash aesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	vec separated(t*n);
	uint64_t activeParent = 0;
	uint8_t choice;
	uint64_t offset;

	for (uint64_t d = 0, w = 1; d < depth; d++, w *= 2) {
		aesHash.expand(tree, separated, tree, w*t); // implement inplace mode
		for (uint64_t i = 0; i < t; i++) {
			// sum in separated
			choice = b[t*(depth-1)+d];
			offset = (t*w/2) * choice + (i*w/2) + activeParent;
			cudaMemcpy(separated.data(offset), cSum.data(d, i), sizeof(blk), cudaMemcpyDeviceToDevice);
			if (d+1 == depth) {
				offset = (w / 2) * (1-choice) + activeParent;
				cudaMemcpy(separated.data(offset), cSum.data(d+1, i), sizeof(blk), cudaMemcpyDeviceToDevice);
			}
		}

		separated.sum(2*t, w/2);

		for (uint64_t i = 0; i < t; i++) {
			// copy into interleaved
			offset = 2 * activeParent + choice;
			cudaMemcpy(tree.data(offset), separated.data(t*choice+i), sizeof(blk), cudaMemcpyDeviceToDevice);
			if (d == depth-1) {
				offset = 2 * activeParent + (1-choice);
				cudaMemcpy(tree.data(offset), separated.data(t*(1-choice)+i), sizeof(blk), cudaMemcpyDeviceToDevice);
			}
			activeParent *= 2;
			activeParent += 1 - choice;
		}
	}

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		printf("spcot_recver: %s\n", cudaGetErrorString(err));
}

void cuda_lpn_f2_compute(int d, int n, int k, uint32_t *key, vec &nn, blk *kk) {
	blk *r_in, *r_out;
	cudaMalloc(&r_in, (d * n / 4) * sizeof(*r_in));
	cudaMalloc(&r_out, (d * n / 4) * sizeof(*r_out));

	uint32_t *key_d;
	cudaMalloc(&key_d, 11 * AES_KEYLEN);
	cudaMemcpy(key_d, key, 11 * AES_KEYLEN, cudaMemcpyHostToDevice);

	dim3 grid(n/4/1024, d);
	make_block<<<grid, 1024>>>(r_in);
	aesEncrypt128<<<d*n/AES_BSIZE, AES_BSIZE>>>(key_d, (uint32_t*)r_out, (uint32_t*)r_in);
	lpn_single_row<<<n / 1024, 1024>>>((uint32_t*)r_out, d, k, nn.data(), kk);

	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
		printf("lpn: %s\n", cudaGetErrorString(err));
}
