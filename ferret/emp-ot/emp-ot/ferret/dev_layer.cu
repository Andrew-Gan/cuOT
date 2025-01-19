#include "aes_op.h"
#include "pprf.h"
#include "dev_layer.h"
#include <cmath>
#include "gpu_ops.h"
#include <cusparse.h>

// optimization options
#define USE_BITSHIFT_SPARSE_MATMUL
#define USE_END_TO_END_PRIMAL_LPN

uint32_t k0_blk[4] = {3242342};
uint32_t k1_blk[4] = {8993849};

void cuda_setdev(int gpu) {
  cudaSetDevice(gpu);
}

void cuda_ipc_open_mem_handle(void **ptr, uint8_t *handle) {
  cudaIpcOpenMemHandle(ptr, *(cudaIpcMemHandle_t*)handle, cudaIpcMemLazyEnablePeerAccess);
  CHECK_CUDA("cuda_ipc_open_mem_handle");
}

void cuda_ipc_close_mem_handle(void *ptr) {
  cudaIpcCloseMemHandle(ptr);
  CHECK_CUDA("cuda_ipc_close_mem_handle");
}

__device__
void blk_xor(blk *a, blk *b) {
  for (int i = 0; i < 4; i++) {
    a->data_32[i] ^= b->data_32[i];
  }
}

void cuda_mpcot_sender(Mat &expanded, Mat &buffer, Mat &sep, blk *lSum_h,
  blk *rSum_h, blk *secret_sum, int t, int depth, blk *delta) {

  blk *delta_d;
  cudaMalloc(&delta_d, sizeof(blk));
  cudaMemcpy(delta_d, delta, sizeof(blk), cudaMemcpyHostToDevice);
  Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  Mat *input = &buffer, *output = &expanded;

  blk seed;
  seed.data_32[0] = rand();
  seed.data_32[1] = rand();
  seed.data_32[2] = rand();
  seed.data_32[3] = rand();
  output->clear();
  output->set(seed, {0});

  for (int d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    std::swap(input, output);
    aesExpand.expand(*input, *output, sep, t * inWidth);
    sep.sum(2 * t, inWidth);
    cudaMemcpy2D(lSum_h+d, depth*sizeof(blk), sep.data(), sizeof(blk),
      sizeof(blk), t, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(rSum_h+d, depth*sizeof(blk), sep.data({(uint64_t)t}),
      sizeof(blk), sizeof(blk), t, cudaMemcpyDeviceToHost);
  }

  if (&expanded != output)
    expanded = *output;
  else
    buffer = *output;

  buffer.sum(t, 1UL << depth);
  uint64_t bufferSize = buffer.size_bytes();
  buffer.xor_scalar(delta_d, t);
  cudaMemcpy(secret_sum, buffer.data(), t*sizeof(blk), cudaMemcpyDeviceToHost);
  cudaFree(delta_d);

  CHECK_CUDA("cuda_mpcot_sender");
}

__global__
void fill_punc_tree(blk *cSum_d, uint64_t outWidth, uint64_t *activeParent,
	bool *choice, blk *puncSum_d, blk *layer, int numTree, int d, int depth) {
		
	uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTree) return;
	uint8_t c = choice[t * depth + d] ? 1 : 0;
	uint64_t fillIndex = t * outWidth + 2 * activeParent[t] + c;
	blk val = layer[fillIndex];
	blk_xor(&val, &cSum_d[t * depth + d]);
	blk_xor(&val, &puncSum_d[c * numTree + t]);
	layer[fillIndex] = val;
	activeParent[t] = 2 * activeParent[t] + (1-c);
}

__global__
void fill_final_punc_tree(uint64_t *activeParent, blk *secret_sum, blk *layer,
  uint64_t numTree, uint64_t treeWidth) {
  
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTree) return;
  uint64_t fillIndex = t * treeWidth + activeParent[t];
  layer[fillIndex] = secret_sum[t];
}

void cuda_mpcot_recver(Mat &expanded, Mat &buffer, Mat &sep, blk *cSum_h,
  blk *secret_sum, int t, int depth, bool *choices_h) {

  blk *secret_sum_d;
  cudaMalloc(&secret_sum_d, t * sizeof(blk));
  Mat cSum_d({(uint64_t)t, (uint64_t)depth});
  cSum_d.read_from_cpu(cSum_h);
  Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  Mat *input = &buffer, *output = &expanded;
  uint64_t *activeParent;
  cudaMalloc(&activeParent, t * sizeof(uint64_t));
  cudaMemset(activeParent, 0, t * sizeof(uint64_t));
  bool *choices_d;
  cudaMalloc(&choices_d, t * depth * sizeof(bool));
  cudaMemcpy(choices_d, choices_h, t * depth * sizeof(bool), cudaMemcpyHostToDevice);
  int block = std::min(t, 1024);
  int grid = (t + block - 1) / block;

  for (int d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    std::swap(input, output);
    aesExpand.expand(*input, *output, sep, t * inWidth);
    sep.sum(2 * t, inWidth);
    fill_punc_tree<<<grid, block>>>(cSum_d.data(), 2*inWidth,
      activeParent, choices_d, sep.data(), output->data(), t, d, depth);
  }

  if (&expanded != output)
    expanded = *output;
  else
    buffer = *output;

  fill_final_punc_tree<<<grid, block>>>(activeParent, secret_sum_d,
    buffer.data(), t, 1UL << depth);
  buffer.sum(t, 1UL << depth);
  fill_final_punc_tree<<<grid, block>>>(activeParent, buffer.data(),
    expanded.data(), t, 1UL << depth);

  cudaFree(choices_d);
  cudaFree(activeParent);
  cudaFree(secret_sum_d);
  CHECK_CUDA("cuda_mpcot_recver");
}

__global__
void primal_lpn_row(uint32_t *r, int64_t d, int k, blk *nn, const blk *kk, uint64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  blk tmp_nn = nn[i];
  for (int j = 0; j < d; j++) {
    blk tmp_kk = kk[r[i*d+j] % k];
    tmp_nn.data_32[0] ^= tmp_kk.data_32[0];
    tmp_nn.data_32[1] ^= tmp_kk.data_32[1];
    tmp_nn.data_32[2] ^= tmp_kk.data_32[2];
    tmp_nn.data_32[3] ^= tmp_kk.data_32[3];
  }
  nn[i] = tmp_nn;
}

void cuda_primal_lpn(Mat &pubMats, int64_t d, int64_t n, int k,
  uint32_t *key, Mat &nn, const blk *kk) {

  uint64_t numBlocks = (pubMats.size() + 1023) / 1024;
  make_block<<<numBlocks, 1024>>>(pubMats.data(), pubMats.size());

  Aes aes(key);
  aes.encrypt(pubMats);

  uint32_t *randMat = (uint32_t*)pubMats.data();
  numBlocks = (n + 1023) / 1024;
  primal_lpn_row<<<numBlocks, 1024>>>(randMat, d, k, nn.data(), kk, n);

  CHECK_CUDA("cuda_primal_lpn");
}

__global__
void _cuda_online_sender(const bool *bo, const blk *ch, blk *data, uint64_t length) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    for (int j = 0; j < 4; j++)
      data[i].data_32[j] ^= ch[bo[i]].data_32[j];
  }
}

void cuda_online_sender(const GPUdata &bo, const Mat &ch, Mat &data, int64_t length) {
  uint64_t numBlock = (length-1) / 1024 + 1;
  _cuda_online_sender<<<numBlock, 1024>>>((bool*)bo.data(), ch.data(), data.data(), length);
  CHECK_CUDA("cuda_online_sender");
}

__global__
void _cuda_online_recver(bool *bo, const bool *b, const blk *data, uint64_t length) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length)
    bo[i] = b[i] ^ *(char*)(data+i) & 1;
}

void cuda_online_recver(GPUdata &bo, void *bo_other, GPUdata &b,
  const Mat &data, int64_t length) {

  uint64_t numBlock = (length-1) / 1024 + 1;
  _cuda_online_recver<<<numBlock, 1024>>>((bool*)bo.data(), (bool*)b.data(),
    data.data(), length);
  int dev = 0;
  cudaGetDevice(&dev);
  int otherDev = dev < 4 ? dev+4 : dev-4;
  cudaMemcpyPeer(bo_other, otherDev, bo.data(), dev, length);
  CHECK_CUDA("cuda_online_recver");
}
