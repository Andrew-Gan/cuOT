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
    a->data[i] ^= b->data[i];
  }
}

void cuda_mpcot_sender(Mat &expanded, Mat &buffer, Mat &sep, blk *lSum_h,
  blk *rSum_h, blk *secret_sum, int t, int depth, blk *delta) {

  blk *delta_d;
  cudaMalloc(&delta_d, sizeof(blk));
  cudaMemcpy(delta_d, delta, sizeof(blk), cudaMemcpyHostToDevice);
  Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  Mat *input = &buffer, *output = &expanded;
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
	blk_xor(&val, &cSum_d[t]);
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
  blk *secret_sum, int t, int depth, bool *choices) {

  bool *b = choices;
  blk *secret_sum_d;
  cudaMalloc(&secret_sum_d, t * sizeof(blk));
  Mat cSum_d({(uint64_t)depth, (uint64_t)t});
  cudaMemcpy(cSum_d.data(), cSum_h, cSum_d.size_bytes(), cudaMemcpyHostToDevice);
  Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  Mat *input = &buffer, *output = &expanded;
  uint64_t *activeParent;
  cudaMalloc(&activeParent, t * sizeof(uint64_t));
  cudaMemset(activeParent, 0, t * sizeof(uint64_t));
  bool *choice;
  cudaMalloc(&choice, t * depth * sizeof(bool));
  cudaMemcpy(choice, b, t * depth * sizeof(bool), cudaMemcpyHostToDevice);
  uint64_t inWidth = 1;
  int block = std::min(t, 1024);
  int grid = (t + block - 1) / block;

  for (int d = 0; d < depth; d++) {
    std::swap(input, output);
    aesExpand.expand(*input, *output, sep, t * inWidth);
    sep.sum(2 * t, inWidth);
    fill_punc_tree<<<grid, block>>>(cSum_d.data({(uint64_t)d, 0}), 2*inWidth,
      activeParent, choice, sep.data(), expanded.data(), t, d, depth);
    inWidth *= 2;
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

  cudaFree(choice);
  cudaFree(activeParent);
  cudaFree(secret_sum_d);
  CHECK_CUDA("cuda_mpcot_recver");
}

#ifdef USE_END_TO_END_PRIMAL_LPN // use end to end primal LPN

__global__
void primal_lpn_row(uint32_t *r, int64_t d, int k, blk *nn, const blk *kk, uint64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  blk tmp_nn = nn[i];
  for (int j = 0; j < d; j++) {
    blk tmp_kk = kk[r[i*d+j] % k];
    tmp_nn.data[0] ^= tmp_kk.data[0];
    tmp_nn.data[1] ^= tmp_kk.data[1];
    tmp_nn.data[2] ^= tmp_kk.data[2];
    tmp_nn.data[3] ^= tmp_kk.data[3];
  }
  nn[i] = tmp_nn;
}

#elif defined(USE_BITSHIFT_SPARSE_MATMUL)

__global__
void matmul_row(uint32_t *r, int64_t d, int k, blk *res, const blk *kk, uint64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  blk tmp;
  memset(&tmp, 0, sizeof(tmp));
  for (int j = 0; j < d; j++) {
    for (int u = 0; u < 4; u++)
      tmp.data[u] ^= kk[r[i*d+j] % k].data[u];
  }
  res[i] = tmp;
}

#else


#else // use cusparse for (sparse matrix * dense vector + dense vector)

__global__
void extract_bit_from_blk(const blk *b, uint8_t *u, uint64_t i, int n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  u[x] = (b[x].data[i / 32] >> (i % 32)) & 0b1;
}

__global__
void extract_bit_from_blk(const blk *b, uint32_t *u, uint64_t i, int n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  u[x] = (b[x].data[i / 32] >> (i % 32)) & 0b1;
}

__global__
void store_bit_into_blk(const uint32_t *u, blk *b, uint64_t i, int n) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  b[x].data[i / 32] |= (u[x] % 2) << (i % 32);
}

void cusparse_primal_lpn(uint32_t *randMat, int d, int n, int k, blk *nn, const blk *kk) {
  static cusparseHandle_t handle = nullptr;
  static int *hA_csrOffsets;
  static int *dA_csrOffsets;
  static uint8_t *hA_values;
  static uint8_t *dA_values;
  static int curr_k = 0;
  static int curr_n = 0;
  static uint8_t *dB_values = nullptr;
  static uint32_t *dC_values = nullptr;

  if (handle == nullptr)
    cusparseCreate(&handle);

  if (curr_k < k || curr_n < n) {
    hA_csrOffsets = new int[n+1];
    hA_values = new uint8_t[d*n];
    for (int i = 0; i < n+1; i++) hA_csrOffsets[i] = 10 * i;
    for (int i = 0; i < d*n; i++) hA_values[i] = 1;
    cudaMalloc(&dA_csrOffsets, (n + 1) * sizeof(int));
    cudaMalloc(&dA_values, d * n * sizeof(uint8_t));
    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, d * n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if (dB_values) cudaFree(dB_values);
    cudaMalloc(&dB_values, k * sizeof(*dB_values));
    if (dC_values) cudaFree(dC_values);
    cudaMalloc(&dC_values, n * sizeof(*dC_values));
    curr_k = k;
    curr_n = n;
  }

  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, n, k, n*d, dA_csrOffsets, randMat, dA_values,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I);

  static GPUdata dBuffer;
  cusparseDnVecDescr_t vecB;
  cusparseDnVecDescr_t vecC;
  int alpha = 1, beta = 1;

  for (int bit = 0; bit < 128; bit++) {
    // cast secret vector of length k
    int blk = std::min(1024, k);
    int grid = (k + blk - 1) / k;
    extract_bit_from_blk<<<grid, blk>>>(kk, dB_values, bit, k);
    cusparseCreateDnVec(&vecB, k, dB_values, CUDA_R_8I);

    // cast noise vector of length n
    blk = std::min(1024, n);
    grid = (n + blk - 1) / n;
    extract_bit_from_blk<<<grid, blk>>>(nn, dC_values, bit, n);
    cusparseCreateDnVec(&vecC, n, dC_values, CUDA_R_32I);

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      matA, vecB, &beta, vecC, CUDA_R_32I, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    dBuffer.resize(bufferSize);

    // execute primal LPN
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB,
      &beta, vecC, CUDA_R_32I, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer.data());
    store_bit_into_blk<<<grid, blk>>>(dC_values, nn, bit, n);
  }

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecB);
  cusparseDestroyDnVec(vecC);
}

#endif // USE_END_TO_END_PRIMAL_LPN

void cuda_primal_lpn(Mat &pubMats, int64_t d, int64_t n, int k,
  uint32_t *key, Mat &nn, const blk *kk) {

  uint64_t numBlocks = (pubMats.size() + 1023) / 1024;
  make_block<<<numBlocks, 1024>>>(pubMats.data(), pubMats.size());

  Aes aes(key);
  aes.encrypt(pubMats);
  uint32_t *randMat = (uint32_t*)pubMats.data();

#ifdef USE_END_TO_END_PRIMAL_LPN
  numBlocks = (n + 1023) / 1024;
  primal_lpn_row<<<numBlocks, 1024>>>(randMat, d, k, nn.data(), kk, n);
#elif defined(USE_BITSHIFT_SPARSE_MATMUL)
  static Mat res;
  res.resize({(uint64_t)n});
  numBlocks = (n + 1023) / 1024;
  matmul_row<<<numBlocks, 1024>>>(randMat, d, k, res.data(), kk, n);
  nn ^= res;
#else
  pubMats.resize({pubMats.size() + 1});
  randMat = (uint32_t*)pubMats.data();
  cusparse_primal_lpn(randMat, d, n, k, nn.data(), kk);
#endif

  CHECK_CUDA("cuda_primal_lpn");
}

__global__
void _cuda_online_sender(const bool *bo, const blk *ch, blk *data, uint64_t length) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    for (int j = 0; j < 4; j++)
      data[i].data[j] ^= ch[bo[i]].data[j];
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
