#include "dev_layer.h"
#include "aes_op.h"

__global__
void make_block(blk *blocks) {
	int x = blockDim.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
  blocks[i*x+j].data[0] = 4*i;
  blocks[i*x+j].data[2] = j;
}

void LpnF2_LpnF2_dev(uint32_t *rdKey, Mat &pubMat) {
  uint32_t* key;
  uint64_t keySize = 11 * AES_KEYLEN;
  cudaMalloc(&key, keySize);
  cudaMemcpy(key, rdKey, keySize, cudaMemcpyHostToDevice);
  dim3 grid(pubMat.dim(1), pubMat.dim(0) / 1024);
  dim3 block(1, 1024);
  make_block<<<grid, block>>>(pubMat.data());
  uint64_t grid2 = 4 * pubMat.dim(0) * pubMat.dim(1) / AES_BSIZE;
  aesEncrypt128<<<grid2, AES_BSIZE>>>(key, (uint32_t*)pubMat.data());
  cudaDeviceSynchronize();
  cudaFree(key);
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

void SPCOT_recver_compute_dev(uint64_t tree_n, Mat &cSum, uint64_t inWidth,
  uint64_t *activeParent, Mat &separated, Span &tree, uint64_t depth,
  uint64_t d, bool *choice) {
  int block = std::min(tree_n, 1024UL);
  int grid = (tree_n + block - 1) / block;
  fill_punc_tree<<<grid, block>>>(cSum.data({d, 0}), 2*inWidth, activeParent,
    choice+(d*tree_n), separated.data(), tree.data(), tree_n);
  if (d == depth-2) {
    fill_punc_tree<<<grid, block>>>(cSum.data({d, tree_n}),
    2*inWidth, activeParent, choice+(d*tree_n), separated.data(),
    tree.data(), tree_n);
  }
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

void LpnF2_encode_dev(Mat &pubMat, uint64_t n, uint64_t k, uint64_t d, Span &nn, Span &kk) {
  lpn_single_row<<<n / 1024, 1024>>>((uint32_t*)pubMat.data(), d, k, nn.data(), kk.data());
    cudaDeviceSynchronize();
}

void set_dev(int dev) {
  cudaSetDevice(dev);
}

void malloc_dev(void **mem, size_t size) {
  cudaMalloc(mem, size);
}

void free_dev(void *mem) {
  cudaFree(mem);
}

void memset_dev(void *des, int val, size_t size) {
  cudaError_t err = cudaMemset(des, val, size);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
}

void memcpy_H2D_dev(void *des, void *src, size_t size) {
  cudaError_t err = cudaMemcpy(des, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
}

void memcpy_D2H_dev(void *des, void *src, size_t size) {
  cudaError_t err = cudaMemcpy(des, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
}

void memcpy_D2D_dev(void *des, void *src, size_t size) {
  cudaError_t err = cudaMemcpy(des, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
}

void sync_dev() {
  cudaDeviceSynchronize();
}