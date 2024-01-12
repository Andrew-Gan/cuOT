#include "aes_op.h"
#include "expand.h"
#include "cuda_layer.h"
#include "gpu_tests.h"
#include <cstdio>
#include <stdexcept>

__global__
void make_block(blk *blocks) {
	int x = blockDim.x;
    int y = blockDim.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
    blocks[i*y+j*x+k].data[0] = 4*j;
    blocks[i*y+j*x+k].data[2] = k;
}

__global__
void lpn_single_row(uint32_t *r, int d, int k, blk *nn, blk *kk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blk tmp_nn = nn[i];
	blk tmp_kk;
    for (int j = 0; j < d; j++) {
        tmp_kk = kk[r[i*d+j] % k];
        tmp_nn.data[0] ^= tmp_kk.data[0];
        tmp_nn.data[1] ^= tmp_kk.data[1];
        tmp_nn.data[2] ^= tmp_kk.data[2];
        tmp_nn.data[3] ^= tmp_kk.data[3];
    }
    nn[i] = tmp_nn;
}

void cuda_init(int party) {
	if (party == 1) cudaSetDevice(0);
	else if (party == 2) cudaSetDevice(1);
	cudaFree(0);
}

void cuda_malloc(void **ptr, size_t n) {
	cudaMalloc(ptr, n);
}

void cuda_memcpy(void *dest, void *src, size_t n, cudaMemcpy_t type) {
	cudaMemcpy(dest, src, n, (cudaMemcpyKind)type);
}

void cuda_free(void *ptr) {
	cudaFree(ptr);
}

void cuda_spcot_sender_compute(Span &tree, int t, int depth, Mat &lSum, Mat &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Vec separated(tree.size());

	for (uint64_t d = 0, w = 1; d < depth-1; d++, w *= 2) {
		aesExpand.expand(tree, separated, w*t);
		separated.sum(2*t, w);
		cudaMemcpy(lSum.data({d, 0}), separated.data(0), t*sizeof(blk), cudaMemcpyDeviceToDevice);
		cudaMemcpy(rSum.data({d, 0}), separated.data(t), t*sizeof(blk), cudaMemcpyDeviceToDevice);
	}

	check_call("spcot_sender\n");
}

void cuda_spcot_recver_compute(Span &tree, int t, int depth, Mat &cSum, bool *b) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Vec separated(tree.size());
	uint64_t *activeParent = new uint64_t[t]();
	uint8_t choice;
	uint64_t offset;

	for (uint64_t d = 0, w = 1; d < depth-1; d++, w *= 2) {
		aesExpand.expand(tree, separated, w*t);
		for (uint64_t i = 0; i < t; i++) {
			// sum in separated
			choice = b[d*t+i];
			offset = choice * (t*w) + (i*w) + activeParent[i];
			cudaMemcpy(separated.data(offset), cSum.data({d, i}), sizeof(blk), cudaMemcpyDeviceToDevice);
			if (d == depth-2) {
				offset = (t*w/2) * (1-choice) + (i*w/2) + activeParent[i];
				cudaMemcpy(separated.data(offset), cSum.data({d, i}), sizeof(blk), cudaMemcpyDeviceToDevice);
			}
		}

		separated.sum(2*t, w/2);

		for (uint64_t i = 0; i < t; i++) {
			offset = 2 * activeParent[i] + choice;
			cudaMemcpy(tree.data(offset), separated.data(t*choice+i), sizeof(blk), cudaMemcpyDeviceToDevice);
			if (d == depth-2) {
				offset = 2 * activeParent[i] + (1-choice);
				cudaMemcpy(tree.data(offset), separated.data(t*(1-choice)+i), sizeof(blk), cudaMemcpyDeviceToDevice);
			}
			activeParent[i] *= 2;
			activeParent[i] += 1 - choice;
		}
	}

	delete[] activeParent;
	check_call("spcot_recver\n");
}

void cuda_gen_matrices(Mat &pubMat, uint32_t *key) {
	dim3 grid(pubMat.dim(2), pubMat.dim(1) / 1024, pubMat.dim(0));
	dim3 block(1, 1024, 1);
	make_block<<<grid, block>>>(pubMat.data());
	grid = dim3(4*pubMat.dim(1)*pubMat.dim(2)/AES_BSIZE, pubMat.dim(0));
	aesEncrypt128<<<grid, AES_BSIZE>>>(key, (uint32_t*)pubMat.data());
	
	check_call("cuda_gen_matrices\n");
}

void cuda_lpn_f2_compute(blk *pubMat, int d, int n, int k, Span &nn, Span &kk) {
	lpn_single_row<<<n/1024, 1024>>>((uint32_t*)pubMat, d, k, nn.data(), kk.data());

	check_call("cuda_lpn_f2_compute\n");
}
