#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) :
  SilentOT(config), fullVector(2 * numOT) {

  expandEvents.resize(depth);
  for (auto &event : expandEvents) {
    cudaEventCreate(&event);
  }

  // pairing
  silentOTSenders[config.id] = this;
  while(silentOTRecvers[config.id] == nullptr);
  other = silentOTRecvers[config.id];

  // seed expansion init
  blk buff;
  for (int i = 0; i < 4; i++)
    buff.data[i] = rand();
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &buff, sizeof(*delta), cudaMemcpyHostToDevice);
  for (int t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++)
      buff.data[i] = rand();
    fullVector.set(t, buff);
  }

  // lpn init
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Sender, 2 * numOT, numOT);
  }
}

void SilentOTSender::base_ot() {
  Log::mem(Sender, BaseOT);

  std::vector<std::future<std::array<Vec, 2>>> workers;
  for (int d = 0; d < depth; d++) {
    workers.push_back(std::async([d, this]() {
      switch (mConfig.baseOT) {
        case SimplestOT_t: return SimplestOT(Sender, d, mConfig.nTree).send();
      }
      return std::array<Vec, 2>();
    }));
  }

  for (auto &worker : workers) {
    std::array<Vec, 2> res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
  leftHash.push_back(leftHash.back());
  rightHash.push_back(rightHash.back());

  Log::mem(Sender, BaseOT);
}

void SilentOTSender::pprf_expand() {
  Log::mem(Sender, SeedExp);

  Expand *expander;
  switch (mConfig.expander) {
    case AesExpand_t:
      expander = new AesExpand((uint8_t*)mConfig.leftKey, (uint8_t*)mConfig.rightKey);
  }

  Log::mem(Sender, SeedExp);

  Vec separated(2 * numOT);
  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    expander->expand(fullVector, separated, mConfig.nTree * inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);
    leftHash.at(d).xor_d(separated, 0);
    rightHash.at(d).xor_d(separated, mConfig.nTree);

    if (d == depth-1) {
      leftHash.at(d+1).xor_d(separated, mConfig.nTree);
      leftHash.at(d+1).xor_scalar(delta);
      rightHash.at(d+1).xor_d(separated, 0);
      rightHash.at(d+1).xor_scalar(delta);
    }

    cudaEventRecord(expandEvents.at(d));
  }

  other->expandReady = true;
  Log::mem(Sender, SeedExp);

  delete expander;
  cudaDeviceSynchronize();
  
  Log::mem(Sender, SeedExp);
}

void SilentOTSender::lpn_compress() {
  lpn->encode(fullVector);
  cudaDeviceSynchronize();
}
