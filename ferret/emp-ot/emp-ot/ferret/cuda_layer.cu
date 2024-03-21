#include "aes_op.h"
#include "pprf.h"
#include "cuda_layer.h"
#include "gpu_tests.h"
#include <cstdio>
#include <stdexcept>

#define NGPU 1

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

void cuda_spcot_sender_compute(Span *tree, int t, int depth, Mat &lSum, Mat &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	Mat buffer[NGPU];
	Span *input[NGPU];
	Span *output[NGPU];
	int treePerGPU = (t + NGPU - 1) / NGPU;
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Mat separated[NGPU];

	for (int gpu = 0; gpu < NGPU; gpu++) {
    	cudaSetDevice(gpu);
		buffer[gpu].resize({tree.size()});
		separated[gpu]->resize({tree.size()});
		input[gpu] = new Span(buffer[gpu], 0, tree.size());
    	output[gpu] = &tree[gpu];
		for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
			std::swap(input[gpu], output[gpu]);
			aesExpand.expand(*(input[gpu]), *(output[gpu]), separated[gpu], treePerGPU*inWidth);
			separated->sum(2 * treePerGPU, inWidth);
			cudaMemcpyAsync(lSum.data({d, 0}), separated[gpu]->data(0), treePerGPU*sizeof(blk), cudaMemcpyDeviceToDevice);
			cudaMemcpyAsync(rSum.data({d, 0}), separated[gpu]->data(treePerGPU), treePerGPU*sizeof(blk), cudaMemcpyDeviceToDevice);
		}
		tree[gpu] = *output[gpu];
	}

	for (int gpu = 0; gpu < NGPU; gpu++) {
		cudaSetDevice(gpu);
		delete input[gpu];
		cudaDeviceSynchronize();
	}
}

void cuda_spcot_recver_compute(Span *tree, int t, int depth, Mat &cSum, bool *b) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	Mat buffer[NGPU];
	Mat *input[NGPU];
	Mat *output[NGPU];
	int treePerGPU = (t + NGPU - 1) / NGPU;
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Mat separated[NGPU];
	uint64_t *activeParent[NGPU];
	bool *choice_d[NGPU];

	int block = std::min(treePerGPU, 1024);
	int grid = (treePerGPU + block - 1) / block;

	for (int gpu = 0; gpu < NGPU; gpu++) {
		cudaSetDevice(gpu);
		buffer[gpu].resize({tree.size_bytes()});
		separated[gpu]->resize({tree.size_bytes()});
		cudaMalloc(&activeParent[gpu], treePerGPU*sizeof(uint64_t), 0);
		cudaMemsetAsync(activeParent[gpu], 0, treePerGPU*sizeof(uint64_t));
		cudaMalloc(&choice_d[gpu], depth*treePerGPU*sizeof(bool), 0);
		cudaMemcpyAsync(choice_d[gpu], b+gpu*treePerGPU, depth*treePerGPU*sizeof(bool), cudaMemcpyHostToDevice);
		for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
			aesExpand.expand(tree, separated, inWidth*treePerGPU);
			separated->sum(2*treePerGPU, inWidth);
			fill_punc_tree<<<grid, block>>>(cSum.data({d, 0}), 2*inWidth, activeParent[gpu], choice_d[gpu]+(d*treePerGPU),
				separated->data(), tree.data(), treePerGPU);
			if (d == depth-2) {
				fill_punc_tree<<<grid, block>>>(cSum.data({d, treePerGPU}), 2*inWidth, activeParent[gpu], choice_d[gpu]+(d*treePerGPU),
				separated->data(), tree.data(), treePerGPU);
			}
		}
	}

	for (int gpu = 0; gpu < NGPU; gpu++) {
		cudaSetDevice(gpu);
		cudaFree(choice_d[gpu]);
		cudaFree(activeParent[gpu]);
		cudaDeviceSynchronize();
	}
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
