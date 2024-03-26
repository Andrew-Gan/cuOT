#include "aes_op.h"
#include "pprf.h"
#include "cuda_layer.h"
#include "gpu_tests.h"
#include <cstdio>
#include <stdexcept>

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

void cuda_spcot_sender_compute(Span &tree, uint64_t t, int depth, Mat &lSum, Mat &rSum) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	Span *input;
	Span *output;
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Mat buffer({tree.size()});
	Mat separated({tree.size()});

	input = new Span(buffer);
	output = &tree;
	for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
		std::swap(input, output);
		aesExpand.expand(*input, *output, separated, t*inWidth);
		separated.sum(2 * t, inWidth);
		cudaMemcpyAsync(lSum.data({d, 0}), separated.data({0}), t*sizeof(blk), cudaMemcpyDeviceToDevice);
		cudaMemcpyAsync(rSum.data({d, 0}), separated.data({t}), t*sizeof(blk), cudaMemcpyDeviceToDevice);
	}
	if (output != &tree) {
		cudaMemcpy(tree.data(), output->data(), tree.size_bytes(), cudaMemcpyDeviceToDevice);
	}

	delete input;
	cudaDeviceSynchronize();
}

void cuda_spcot_recver_compute(Span &tree, uint64_t t, int depth, Mat &cSum, bool *b) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	Span *input;
	Span *output;
	AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
	Mat buffer({tree.size()});
	Mat separated({tree.size()});
	uint64_t *activeParent;
	bool *choice;

	int block = std::min(t, 1024UL);
	int grid = (t + block - 1) / block;

	cudaMalloc(&activeParent, t*sizeof(uint64_t));
	cudaMalloc(&choice, depth*t);
	cudaMemsetAsync(activeParent, 0, t*sizeof(uint64_t));
	cudaMemcpyAsync(choice, b, depth*t, cudaMemcpyHostToDevice);
	input = new Span(buffer);
	output = &tree;
	for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
		std::swap(input, output);
		aesExpand.expand(*input, *output, separated, t*inWidth);
		separated.sum(2 * t, inWidth);
		fill_punc_tree<<<grid, block>>>(cSum.data({d, 0}), 2*inWidth,
			activeParent, choice+(d*t), separated.data(),
			tree.data(), t);
		if (d == depth-2) {
			fill_punc_tree<<<grid, block>>>(cSum.data({d, t}),
			2*inWidth, activeParent, choice+(d*t), separated.data(),
			tree.data(), t);
		}
	}
	if (output != &tree) {
		cudaMemcpy(tree.data(), output->data(), tree.size_bytes(), cudaMemcpyDeviceToDevice);
	}

	delete input;
	cudaFree(choice);
	cudaFree(activeParent);
	cudaDeviceSynchronize();
}
