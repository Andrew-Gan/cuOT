#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"
#include "gpu_tests.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) :
  SilentOT(config) {

  expandEvents.resize(NGPU);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    if (gpu == 0)
      fullVector[gpu].resize(2 * numOT); // main vector
    else
      fullVector[gpu].resize(2 * numOT / NGPU);
    expandEvents.at(gpu).resize(depth);
    for (cudaEvent_t &e : expandEvents.at(gpu)) {
      cudaEventCreate(&e);
    }

    // pairing
    silentOTSenders[config.id] = this;
    while(silentOTRecvers[config.id] == nullptr);
    other = silentOTRecvers[config.id];

    // seed expansion init
    blk buff = {.data = {
      (uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand(), (uint32_t)rand()
    }};
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
  }

  cudaSetDevice(0);
  
  // lpn init
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Sender, 2 * numOT, numOT);
  }
}

SilentOTSender::~SilentOTSender() {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete expander[gpu];
  }
  delete lpn;
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
      leftHash[gpu].push_back(res[0]);
      rightHash[gpu].push_back(res[1]);
    }
    leftHash[gpu].push_back(leftHash[gpu].back());
    rightHash[gpu].push_back(rightHash[gpu].back());
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
      leftHash[gpu].at(d).xor_d(separated[gpu], 0);
      rightHash[gpu].at(d).xor_d(separated[gpu], treePerGPU);

      if (d == depth-1) {
        leftHash[gpu].at(d+1).xor_d(separated[gpu], treePerGPU);
        leftHash[gpu].at(d+1).xor_scalar(delta[gpu]);
        rightHash[gpu].at(d+1).xor_d(separated[gpu], 0);
        rightHash[gpu].at(d+1).xor_scalar(delta[gpu]);
      }

      cudaEventRecord(expandEvents.at(gpu).at(d));
    }
  }

  other->expandReady = true;
  cudaSetDevice(0);
  Log::mem(Sender, SeedExp);

  for (int gpu = 1; gpu < NGPU; gpu++) {
    uint64_t offset = gpu * fullVector[gpu].size();
    cudaMemcpyPeerAsync(fullVector[0].data(offset), 0,
    fullVector[gpu].data(), gpu, fullVector[gpu].size_bytes());
  }
  cudaDeviceSynchronize();
}

void SilentOTSender::lpn_compress() {
  lpn->encode(fullVector[0]);
  cudaDeviceSynchronize();
}
