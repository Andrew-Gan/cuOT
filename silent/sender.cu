#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) :
  SilentOT(config) {

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    fullVector[gpu].resize(2 * numOT / NGPU);
    std::vector<cudaEvent_t> &events = expandEvents[gpu];
    events.resize(depth);
    for (uint64_t i = 0; i < depth; i++) {
      cudaEventCreate(&events.at(i));
    }

    // pairing
    silentOTSenders[config.id] = this;
    while(silentOTRecvers[config.id] == nullptr);
    other = silentOTRecvers[config.id];

    blk buff;
    for (int i = 0; i < 4; i++) buff.data[i] = rand();
    cudaMalloc(&delta[gpu], sizeof(**delta));
    cudaMemcpy(delta[gpu], &buff, sizeof(**delta), cudaMemcpyHostToDevice);
    for (int t = 0; t < mConfig.nTree / NGPU; t++) {
      for (int i = 0; i < 4; i++) buff.data[i] = rand();
      fullVector[gpu].set(t, buff);
    }
    switch (mConfig.expander) {
      case AesExpand_t:
        expander[gpu] = new AesExpand((uint8_t*)mConfig.leftKey, (uint8_t*)mConfig.rightKey);
    }
    switch (mConfig.compressor) {
      case QuasiCyclic_t:
        lpn[gpu] = new QuasiCyclic(Sender, 2 * numOT / NGPU, numOT / NGPU, NGPU);
    }
  }
  cudaSetDevice(0);
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
  int treePerGPU = mConfig.nTree / NGPU;
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
        m0[gpu].at(d+1).xor_scalar(delta[gpu]);
        m1[gpu].at(d+1).xor_d(separated[gpu], 0);
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
  cudaSetDevice(0);
}

void SilentOTSender::lpn_compress() {
  Log::mem(Sender, LPN);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    lpn[gpu]->encode(fullVector[gpu]);
  }

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);
}
