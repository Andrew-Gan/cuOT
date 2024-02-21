#include "base_ot.h"
#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

SilentOTRecver::SilentOTRecver(SilentOTConfig config) : SilentOT(config) {
  silentOTRecvers[mConfig.id] = this;
  while(silentOTSenders[mConfig.id] == nullptr);
  other = silentOTSenders[mConfig.id];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    puncVector[gpu].resize({numOT / NGPU, 1});
    m0[gpu] = std::vector<Mat>(depth+1, Mat({mConfig.nTree / NGPU}));
    m1[gpu] = std::vector<Mat>(depth+1, Mat({mConfig.nTree / NGPU}));
    cudaMalloc(&activeParent[gpu], mConfig.nTree / NGPU * sizeof(uint64_t));
    cudaMemset(activeParent[gpu], 0, mConfig.nTree / NGPU * sizeof(uint64_t));
    switch (mConfig.expander) {
      case AesExpand_t:
        expander[gpu] = new AesExpand(mConfig.leftKey, mConfig.rightKey);
    }
    switch (mConfig.compressor) {
      case QuasiCyclic_t:
        lpn[gpu] = new QuasiCyclic(Recver, 2 * numOT, numOT, BLOCK_BITS / NGPU);
    }
  }

  cudaSetDevice(mConfig.ngpuAvail-1);
  cudaMalloc(&puncPos, mConfig.nTree * sizeof(uint64_t));
  get_choice_vector();
  lpn[0]->encode_sparse(choiceVector, puncPos, mConfig.nTree);
}

SilentOTRecver::~SilentOTRecver() {
  cudaSetDevice(mConfig.ngpuAvail-1);
  cudaFree(puncPos);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    cudaFree(activeParent[gpu]);
    delete expander[gpu];
    delete lpn[gpu];
  }
}


void SilentOTRecver::base_ot() {
  Log::mem(Recver, BaseOT);
  std::vector<std::future<Mat>> workers[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    for (int d = 0; d < depth; d++) {
      workers[gpu].push_back(std::async([d, gpu, this]() {
        cudaSetDevice(mConfig.ngpuAvail-gpu-1);
        uint64_t tree = mConfig.nTree / NGPU;
        uint64_t choices = mConfig.choices[d] >> (gpu * tree);
        switch (mConfig.baseOT) {
          case SimplestOT_t:
            return SimplestOT(Recver, gpu*this->depth+d, tree).recv(choices);
        }
        return Mat();
      }));
    }
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    for (auto &worker : workers[gpu]) {
      auto res = worker.get();
      mc[gpu].push_back(res);
    }
  }
  Log::mem(Recver, BaseOT);
}

__global__
void choice_bits_to_pos(uint64_t *choiceVector, uint64_t *choiceBits, uint64_t depth) {
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t id = 0;
  for (uint64_t d = 0; d < depth; d++) {
    id *= 2;
    id += 1-(choiceBits[d] >> t & 1);
  }
  choiceVector[t] = id + t * (1 << depth);
}

void SilentOTRecver::get_choice_vector() {
  uint64_t *choices_d;
  cudaMalloc(&choices_d, depth * sizeof(*choices_d));
  cudaMemcpyAsync(choices_d, mConfig.choices, depth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  choice_bits_to_pos<<<1, mConfig.nTree>>>(puncPos, choices_d, depth);
  cudaDeviceSynchronize();
  cudaFree(choices_d);
}

__global__
void fill_tree(blk *leftSum, blk *rightSum, uint64_t outWidth, uint64_t *activeParent,
  uint64_t choice, blk *puncSum, blk *layer, bool finalLayer) {

  uint64_t numTree = gridDim.x * blockDim.x;
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  int c = (choice >> t) & 1;
  blk *fullSum = c == 0 ? leftSum : rightSum;
  uint64_t fillIndex = t * outWidth;
  fillIndex += finalLayer ? activeParent[t] : 2 * activeParent[t] + c;
  blk val = layer[fillIndex];
  uint64_t puncOffset = (finalLayer ? 1-c : c) * numTree + t;
  for (int i = 0; i < 4; i++)
    val.data[i] ^= fullSum[t].data[i] ^ puncSum[puncOffset].data[i];
  layer[fillIndex] = val;
  if (!finalLayer)
    activeParent[t] = 2 * activeParent[t] + (1-c);
}

void SilentOTRecver::pprf_expand() {
  Log::mem(Recver, SeedExp);
  int treePerGPU = (mConfig.nTree + NGPU - 1) / NGPU;
  Mat separated[NGPU];
  Mat buffer[NGPU];
  Mat *input[NGPU];
  Mat *output[NGPU];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    separated[gpu].resize({numOT / NGPU});
    buffer[gpu].resize(puncVector[gpu].dims());
    input[gpu] = &buffer[gpu];
    output[gpu] = &puncVector[gpu];
    for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
      std::swap(input[gpu], output[gpu]);
      expander[gpu]->expand(*(input[gpu]), *(output[gpu]), separated[gpu], treePerGPU*inWidth);
      separated[gpu].sum(2 * treePerGPU, inWidth);
      cudaStreamWaitEvent(0, other->expandEvents[gpu].at(d));

      m0[gpu].at(d) = other->m0[gpu].at(d);
      m1[gpu].at(d) = other->m1[gpu].at(d);
      m0[gpu].at(d).xor_d(mc[gpu].at(d));
      m1[gpu].at(d).xor_d(mc[gpu].at(d));

      fill_tree<<<1, treePerGPU>>>(m0[gpu].at(d).data(), m1[gpu].at(d).data(),
        2 * inWidth, activeParent[gpu], mConfig.choices[d] >> (gpu*treePerGPU),
        separated[gpu].data(), output[gpu]->data(), false);
      
      if (d == depth-1) {
        m0[gpu].at(d+1) = other->m0[gpu].at(d+1);
        m1[gpu].at(d+1) = other->m1[gpu].at(d+1);
        m0[gpu].at(d+1).xor_d(mc[gpu].at(d));
        m1[gpu].at(d+1).xor_d(mc[gpu].at(d));

        fill_tree<<<1, treePerGPU>>>(m0[gpu].at(d+1).data(), m1[gpu].at(d+1).data(),
          2 * inWidth, activeParent[gpu], mConfig.choices[d] >> (gpu*treePerGPU),
          separated[gpu].data(), output[gpu]->data(), true);
      }
    }
  }
  Log::mem(Recver, SeedExp);

  for (int gpu = 0; gpu < NGPU; gpu++) {
    if (&puncVector[gpu] != output[gpu])
      puncVector[gpu] = *(output[gpu]);
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    cudaDeviceSynchronize();
  }
}

void SilentOTRecver::lpn_compress() {
  Log::mem(Recver, LPN);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    puncVector[gpu].bit_transpose();
  }

  Mat b64[NGPU];
  uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
  for (int des = 0; des < NGPU; des++) {
    cudaSetDevice(mConfig.ngpuAvail-des-1);
    b64[des].resize({rowsPerGPU, 2 * numOT / BLOCK_BITS});
    b64[des].clear();
    for (int src = 0; src < NGPU; src++) {
      cudaMemcpy2DPeerAsync(
        b64[des].data({0, src*puncVector[src].dim(1)}), b64[des].dim(1)*sizeof(blk), mConfig.ngpuAvail-des-1,
        puncVector[src].data({des*b64[des].dim(0), 0}), puncVector[src].dim(1)*sizeof(blk), mConfig.ngpuAvail-src-1,
        puncVector[src].dim(1)*sizeof(blk), b64[des].dim(0)
      );
    }
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    lpn[gpu]->encode_dense(b64[gpu]);
  }
  cudaSetDevice(mConfig.ngpuAvail-1);
  puncVector[0].resize({BLOCK_BITS, numOT / BLOCK_BITS});

  uint64_t dWidth = puncVector[0].dim(1)*sizeof(blk);
  uint64_t sWidth = b64[0].dim(1)*sizeof(blk);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaMemcpy2DPeerAsync(
      puncVector[0].data({gpu * rowsPerGPU, 0}), dWidth, mConfig.ngpuAvail-1,
      b64[gpu].data(), sWidth, mConfig.ngpuAvail-gpu-1, dWidth, rowsPerGPU
    );
  }
  puncVector[0].bit_transpose();

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(mConfig.ngpuAvail-gpu-1);
    cudaDeviceSynchronize();
  }
}
