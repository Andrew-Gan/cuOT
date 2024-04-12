#include "aes_op.h"
#include "pprf.h"
#include "dev_layer.h"
#include <cmath>
#include "gpu_ops.h"
#include <curand_kernel.h>
#include <future>

__device__
void blk_xor(blk *a, blk *b) {
  for (int i = 0; i < 4; i++) {
    a->data[i] ^= b->data[i];
  }
}

void cuda_mpcot_sender(Mat *expanded, blk *lSum_h, blk *rSum_h,
  blk *secret_sum, int tree, int depth, blk *Delta_f2k, int ngpu) {
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	uint64_t treePerGPU = (tree + ngpu - 1) / ngpu;
  std::vector<std::future<void>> workers;

	for (int gpu = 0; gpu < ngpu; gpu++) {
    blk *lS = &lSum_h[gpu*treePerGPU*depth];
    blk *rS = &rSum_h[gpu*treePerGPU*depth];
    workers.push_back(std::async([&, gpu, lS, rS](){
      cudaSetDevice(gpu);
      expanded[gpu].resize({treePerGPU * (1UL << depth)});

	    Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
      Mat buffer(expanded[gpu].dims());
      Mat sep(expanded[gpu].dims());
      Mat *input = &buffer, *output = &expanded[gpu];

      for (int d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
        std::swap(input, output);
        aesExpand.expand(*input, *output, sep, treePerGPU * inWidth);
        sep.sum(2 * treePerGPU, inWidth);
        cudaMemcpy2D(lS+d, depth*sizeof(blk), sep.data(), sizeof(blk), sizeof(blk), treePerGPU, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(rS+d, depth*sizeof(blk), sep.data({treePerGPU}), sizeof(blk), sizeof(blk), treePerGPU, cudaMemcpyDeviceToHost);
      }
      if (&expanded[gpu] != output)
        expanded[gpu] = *output;
      else
        buffer = *output;
      buffer.sum(treePerGPU, 1UL << depth);
      buffer.resize({treePerGPU});
      blk *delta;
      cudaMalloc(&delta, sizeof(blk));
      cudaMemcpy(delta, Delta_f2k, sizeof(blk), cudaMemcpyHostToDevice);
      buffer.xor_scalar(delta);
      cudaMemcpy(secret_sum+gpu*treePerGPU, buffer.data(), treePerGPU*sizeof(blk), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      cudaFree(delta);
    }));
	}

	for (int gpu = 0; gpu < ngpu; gpu++) {
		workers.at(gpu).get();
	}
}

__global__
void fill_punc_tree(blk *cSum, uint64_t outWidth, uint64_t *activeParent,
	bool *choice, blk *puncSum, blk *layer, int numTree, int d, int depth) {
		
	uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= numTree) return;
	uint8_t c = choice[t * depth + d] ? 1 : 0;

	uint64_t fillIndex = t * outWidth + 2 * activeParent[t] + c;
	blk val = layer[fillIndex];
	blk_xor(&val, &cSum[t]);
	blk_xor(&val, &puncSum[c * numTree + t]);
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

void cuda_mpcot_recver(Mat *expanded, blk *cSum_h, blk *secret_sum, int tree,
  int depth, bool *choices, int ngpu) {
  // return; // uncomment when benchmarking sender only
  int gpuCount = 0;
  cudaGetDeviceCount(&gpuCount);
	uint32_t k0_blk[4] = {3242342};
	uint32_t k1_blk[4] = {8993849};
	uint64_t treePerGPU = (tree + ngpu - 1) / ngpu;
  std::vector<std::future<void>> workers;
	int block = std::min(treePerGPU, 1024UL);
	int grid = (treePerGPU + block - 1) / block;

	for (int gpu = 0; gpu < ngpu; gpu++) {
    bool *b = choices + gpu * treePerGPU * depth;
    blk *cS = &cSum_h[gpu*treePerGPU*depth];
    workers.push_back(std::async([&, gpu, b, cS, treePerGPU]() {
      cudaSetDevice(gpu);
      blk *secret_sum_d;
      cudaMalloc(&secret_sum_d, treePerGPU * sizeof(blk));
      expanded[gpu].resize({treePerGPU * (1UL << depth)});
      Mat cSum({(uint64_t)depth, (uint64_t)treePerGPU});
      cudaMemcpy(cSum.data(), cS, cSum.size_bytes(), cudaMemcpyHostToDevice);
      Aes aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
      Mat buffer(expanded[gpu].dims());
      Mat sep(expanded[gpu].dims());
      Mat *input = &buffer, *output = &expanded[gpu];
      uint64_t *activeParent;
      cudaMalloc(&activeParent, treePerGPU * sizeof(uint64_t));
      cudaMemset(activeParent, 0, treePerGPU * sizeof(uint64_t));
      bool *choice;
      cudaMalloc(&choice, treePerGPU * depth * sizeof(bool));
      cudaMemcpy(choice, b, treePerGPU * depth * sizeof(bool), cudaMemcpyHostToDevice);
      uint64_t inWidth = 1;
      for (int d = 0; d < depth; d++) {
        std::swap(input, output);
        aesExpand.expand(*input, *output, sep, treePerGPU * inWidth);
        sep.sum(2 * treePerGPU, inWidth);
        fill_punc_tree<<<grid, block>>>(cSum.data({(uint64_t)d, 0}), 2*inWidth, activeParent, choice,
          sep.data(), expanded[gpu].data(), treePerGPU, d, depth);
        inWidth *= 2;
      }
      
      if (&expanded[gpu] != output)
        expanded[gpu] = *output;
      else
        buffer = *output;

      fill_final_punc_tree<<<grid, block>>>(activeParent, secret_sum_d, buffer.data(), treePerGPU, 1UL << depth);
      buffer.sum(treePerGPU, 1UL << depth);
      fill_final_punc_tree<<<grid, block>>>(activeParent, buffer.data(), expanded[gpu].data(), treePerGPU, 1UL << depth);

      cudaFree(choice);
      cudaFree(activeParent);
      cudaDeviceSynchronize();
    }));
	}

	for (int gpu = 0; gpu < ngpu; gpu++) {
		workers.at(gpu).get();
	}
}

void cuda_gen_matrices(Role role, Mat *pubMats, int64_t n, int d, uint32_t *key,
  int ngpu) {
  // if (role == Recver) return; // uncomment when benchmarking sender only
  int gpuCount = 0;
  cudaGetDeviceCount(&gpuCount);
  uint64_t rowsPerGPU = (n + ngpu - 1) / ngpu;
  std::vector<std::future<void>> workers;
  for (int gpu = 0; gpu < ngpu; gpu++) {
    workers.push_back(std::async([&, gpu]() {
      cudaSetDevice(gpu);
      pubMats[gpu].resize({(rowsPerGPU * d + 3) / 4});

      // generate random matrix using aes encryption
      make_block<<<pubMats[gpu].size() / 1024, 1024>>>(pubMats[gpu].data(), gpu * rowsPerGPU);
      Aes aes(key);
      aes.encrypt(pubMats[gpu]);

      cudaDeviceSynchronize();
    }));
  }

  for (int gpu = 0; gpu < ngpu; gpu++) {
		workers.at(gpu).get();
	}
}

__global__
void lpn_single_row(uint32_t *r, int64_t d, int k, blk *nn, blk *kk) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
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

void cuda_primal_lpn(Role role, Mat *pubMats, int64_t d, int64_t n, int k,
  Mat *expanded, blk *nn, blk *kk, int ngpu) {
  // if (role == Recver) return; // uncomment when benchmarking sender only
  int gpuCount = 0;
  cudaGetDeviceCount(&gpuCount);
  uint64_t rowsPerGPU = (n + ngpu - 1) / ngpu;
  std::vector<std::future<void>> workers;
  for (int gpu = 0; gpu < ngpu; gpu++) {
    workers.push_back(std::async([&, gpu, rowsPerGPU]() {
      cudaSetDevice(gpu);
      Mat kk_d({(uint64_t)k});
      cudaMemcpy(kk_d.data(), kk, kk_d.size_bytes(), cudaMemcpyHostToDevice);
      lpn_single_row<<<rowsPerGPU/1024, 1024>>>((uint32_t*)pubMats[gpu].data(),
        d, k, expanded[gpu].data(), kk_d.data());
      cudaMemcpy(nn+gpu*rowsPerGPU, expanded[gpu].data(), expanded[gpu].size_bytes(), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
    }));
  }

  for (int gpu = 0; gpu < ngpu; gpu++) {
		workers.at(gpu).get();
	}
}
