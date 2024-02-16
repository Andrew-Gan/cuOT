#include "aes_op.h"
#include "expand.h"
#include "cuda_layer.h"
#include "gpu_tests.h"
#include <cstdio>
#include <stdexcept>

__global__
void make_block(blk *blocks) {
	int x = blockDim.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    blocks[i*x+j].data[0] = 4*i;
    blocks[i*x+j].data[2] = j;
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

__device__
void blk_xor(blk *a, blk *b) {
  for (int i = 0; i < 4; i++) {
    a->data[i] ^= b->data[i];
  }
}

__global__
void fill_punc_tree(blk *cSum, uint64_t outWidth, uint64_t *activeParent,
	bool *choice, blk *puncSum, blk *layer, int numTree) {
		
	uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTree) return;
	uint8_t c = choice[t];

	uint64_t fillIndex = t * outWidth + 2 * activeParent[t] + c;
	blk val = layer[fillIndex];
	blk_xor(&val, &cSum[t]);
	blk_xor(&val, &puncSum[c * numTree + t]);
	layer[fillIndex] = val;
	activeParent[t] = 2 * activeParent[t] + (1-c);
}

void cuda_spcot_sender_compute(Span &tree, int t, int depth, Mat &lSum, Mat &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Vec separated(tree.size());

	for (uint64_t d = 0, w = 1; d < depth-1; d++, w *= 2) {
		aesExpand.expand(tree, separated, w*t);
		separated.sum(2*t, w);
		cudaMemcpyAsync(lSum.data({d, 0}), separated.data(0), t*sizeof(blk), cudaMemcpyDeviceToDevice);
		cudaMemcpyAsync(rSum.data({d, 0}), separated.data(t), t*sizeof(blk), cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
}

void cuda_spcot_recver_compute(Span &tree, int t, int depth, Mat &cSum, bool *b) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Vec separated(tree.size());
	uint64_t *activeParent;
	cudaMalloc(&activeParent, t*sizeof(uint64_t));
	cudaMemsetAsync(activeParent, 0, t*sizeof(uint64_t));
	bool *choice_d;
	cudaMalloc(&choice_d, depth*t*sizeof(bool));
	cudaMemcpyAsync(choice_d, b, depth*t*sizeof(bool), cudaMemcpyHostToDevice);

	int block = std::min(t, 1024);
	int grid = (t + block - 1) / block;

	for (uint64_t d = 0, w = 1; d < depth-1; d++, w *= 2) {
		aesExpand.expand(tree, separated, w*t);
		separated.sum(2*t, w);
		fill_punc_tree<<<grid, block>>>(cSum.data({d, 0}), 2*w, activeParent, &choice_d[d*t],
			separated.data(), tree.data(), t);
		if (d == depth-2) {
			fill_punc_tree<<<grid, block>>>(cSum.data({d, 0}), 2*w, activeParent, &choice_d[d*t],
			separated.data(), tree.data(), t);
		}
	}

	cudaFreeAsync(choice_d, 0);
	cudaFreeAsync(activeParent, 0);
	cudaDeviceSynchronize();
}

void cuda_gen_matrices(Mat &pubMat, uint32_t *key) {
	dim3 grid(pubMat.dim(1), pubMat.dim(0) / 1024);
	dim3 block(1, 1024);
	make_block<<<grid, block>>>(pubMat.data());
	uint64_t grid2 = 4*pubMat.dim(0)*pubMat.dim(1)/AES_BSIZE;
	aesEncrypt128<<<grid2, AES_BSIZE>>>(key, (uint32_t*)pubMat.data());
	cudaDeviceSynchronize();
}

void cuda_lpn_f2_compute(blk *pubMat, int d, int n, int k, Span &nn, Span &kk) {
	lpn_single_row<<<n/1024, 1024>>>((uint32_t*)pubMat, d, k, nn.data(), kk.data());
	cudaDeviceSynchronize();
}
