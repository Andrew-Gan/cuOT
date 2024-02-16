#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

#include "gpu_tests.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) : SilentOT(config) {
  blk seed_h, delta_h;
  for (int i = 0; i < 4; i++) delta_h.data[i] = rand();

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    fullVector[gpu].resize(2 * numOT / NGPU);
    std::vector<cudaEvent_t> &events = expandEvents[gpu];
    events.resize(depth);
    for (uint64_t i = 0; i < depth; i++) {
      cudaEventCreate(&events.at(i));
    }
    silentOTSenders[config.id] = this;
    while(silentOTRecvers[config.id] == nullptr);
    other = silentOTRecvers[config.id];

    cudaMalloc(&delta[gpu], sizeof(**delta));
    cudaMemcpy(delta[gpu], &delta_h, sizeof(**delta), cudaMemcpyHostToDevice);
    for (int t = 0; t < mConfig.nTree / NGPU; t++) {
      for (int i = 0; i < 4; i++) seed_h.data[i] = rand();
      fullVector[gpu].set(t, seed_h);
    }
    switch (mConfig.expander) {
      case AesExpand_t:
        expander[gpu] = new AesExpand(mConfig.leftKey, mConfig.rightKey);
    }
    switch (mConfig.compressor) {
      case QuasiCyclic_t:
        lpn[gpu] = new QuasiCyclic(Sender, 2*numOT, numOT, BLOCK_BITS/NGPU);
    }
  }
}

SilentOTSender::~SilentOTSender() {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    delete expander[gpu];
    delete lpn[gpu];
    for (uint64_t d = 0; d < depth; d++)
      cudaEventDestroy(expandEvents[gpu].at(d));
    cudaFree(delta[gpu]);
  }
}

void SilentOTSender::base_ot() {
  Log::mem(Sender, BaseOT);
  std::vector<std::future<std::array<Vec, 2>>> workers[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    for (int d = 0; d < depth; d++) {
      workers[gpu].push_back(std::async([d, gpu, this]() {
        cudaSetDevice(gpu);
        switch (mConfig.baseOT) {
          case SimplestOT_t:
            return SimplestOT(Sender, gpu*this->depth+d, mConfig.nTree / NGPU).send();
        }
        return std::array<Vec, 2>();
      }));
    }
  }

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    for (auto &worker : workers[gpu]) {
      std::array<Vec, 2> res = worker.get();
      m0[gpu].push_back(res[0]);
      m1[gpu].push_back(res[1]);
    }
    m0[gpu].push_back(m0[gpu].back());
    m1[gpu].push_back(m1[gpu].back());
  }
  Log::mem(Sender, BaseOT);
}

void SilentOTSender::pprf_expand() {
  Log::mem(Sender, SeedExp);
  int treePerGPU = (mConfig.nTree + NGPU - 1) / NGPU;
  Vec separated[NGPU];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    separated[gpu] = Vec(2 * numOT / NGPU);
    for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
      expander[gpu]->expand(fullVector[gpu], separated[gpu], treePerGPU*inWidth);
      separated[gpu].sum(2 * treePerGPU, inWidth);
      m0[gpu].at(d).xor_d(separated[gpu], 0);
      m1[gpu].at(d).xor_d(separated[gpu], treePerGPU);

      if (d == depth-1) {
        m0[gpu].at(d+1).xor_d(separated[gpu], treePerGPU);
        m1[gpu].at(d+1).xor_d(separated[gpu], 0);
        m0[gpu].at(d+1).xor_scalar(delta[gpu]);
        m1[gpu].at(d+1).xor_scalar(delta[gpu]);
      }
      cudaEventRecord(expandEvents[gpu].at(d));
    }
  }
  other->expandReady = true;
  Log::mem(Sender, SeedExp);

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
}

void SilentOTSender::lpn_compress() {
  Log::mem(Sender, LPN);
  Mat *tmp = new Mat[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    tmp[gpu].resize({numOT / NGPU, 1});
    tmp[gpu].load(fullVector[gpu].data(), numOT / NGPU * sizeof(OTblock));
    tmp[gpu].bit_transpose();
  }
  Mat b64[NGPU];
  uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
  for (int des = 0; des < NGPU; des++) {
    cudaSetDevice(des);
    b64[des].resize({rowsPerGPU, numOT / BLOCK_BITS});
    b64[des].clear();
    for (int src = 0; src < NGPU; src++) {
      cudaMemcpy2DPeerAsync(
        b64[des].data({0, src*tmp[src].dim(1)}), b64[des].dim(1)*sizeof(blk), des,
        tmp[src].data({des*b64[des].dim(0), 0}), tmp[src].dim(1)*sizeof(blk), src,
        tmp[src].dim(1)*sizeof(blk), b64[des].dim(0)
      );
    }
  }
  delete[] tmp;
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    lpn[gpu]->encode_dense(b64[gpu]);
  }
  cudaSetDevice(0);
  b64[0].resize({BLOCK_BITS, b64[0].dim(1)});
  for (int gpu = 1; gpu < NGPU; gpu++) {
    cudaMemcpyPeerAsync(
      b64[0].data({gpu * rowsPerGPU, 0}), 0,
      b64[gpu].data(), gpu, b64[gpu].size_bytes()
    );
  }
  
  b64[0].bit_transpose();
  fullVector[0].resize(b64[0].dim(0));
  fullVector[0].load(b64[0].data());

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
}
