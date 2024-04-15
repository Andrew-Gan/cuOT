#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

blk* SilentOTSender::m0_h = nullptr;
blk* SilentOTSender::m1_h = nullptr;
std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentConfig config) : SilentOT(config) {
  blk seed_h, delta_h;
  mRole = Sender;
  cudaSetDevice(mConfig.id);
  silentOTSenders[mConfig.id] = this;
  for (int i = 0; i < 4; i++)
    delta_h.data[i] = rand();
  
  m0.resize({mDepth+1, mConfig.nTree});
  m1.resize({mDepth+1, mConfig.nTree});

  fullVector = new Mat({numOT, 1});
  buffer = new Mat(fullVector->dims());
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &delta_h, sizeof(*delta), cudaMemcpyHostToDevice);
  for (uint64_t t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) seed_h.data[i] = rand();
    fullVector->set(seed_h, {t, 0});
  }
  separated.resize({numOT});
  switch (mConfig.pprf) {
    case Aes_t:
      expander = new Aes(mConfig.leftKey, mConfig.rightKey);
  }

  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Sender, 2 * numOT, numOT, BLOCK_BITS / mConfig.gpuPerParty);
  }

  if (mConfig.id == 0) {
    SilentOTSender::m0_h = new blk[(mDepth+1) * mConfig.nTree];
    SilentOTSender::m1_h = new blk[(mDepth+1) * mConfig.nTree];
  }
}

SilentOTSender::~SilentOTSender() {
  cudaSetDevice(mConfig.id);
  delete fullVector;
  delete buffer;
  delete expander;
  delete lpn;
  if (mConfig.id == 0) {
    delete[] SilentOTSender::m0_h;
    delete[] SilentOTSender::m1_h;
  }
  cudaFree(delta);
  silentOTSenders[mConfig.id] = nullptr;
}

void SilentOTSender::base_ot() {
  cudaSetDevice(mConfig.id);
  std::vector<std::future<void>> workers;
  for (uint64_t d = 0; d < mDepth; d++) {
    workers.push_back(std::async([d, this](){
      SimplestOT bOT(Sender, d, mConfig.nTree);
      bOT.send(SilentOTSender::m0_h+d*mConfig.nTree, SilentOTSender::m1_h+d*mConfig.nTree);
    }));
  }
  for (auto &t : workers) {
    t.get();
  }
}

void SilentOTSender::seed_expand() {
  cudaSetDevice(mConfig.id);
  Log::mem(Sender, SeedExp);

  cudaMemcpy(m0.data(), SilentOTSender::m0_h, m0.size_bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(m1.data(), SilentOTSender::m1_h, m1.size_bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(m0.data({mDepth, 0}), m0.data({mDepth-1, 0}), m0.dim(1)*sizeof(blk), cudaMemcpyDeviceToDevice);
  cudaMemcpy(m1.data({mDepth, 0}), m1.data({mDepth-1, 0}), m1.dim(1)*sizeof(blk), cudaMemcpyDeviceToDevice);

  Mat *input = buffer;
  Mat *output = fullVector;
  uint64_t numBytes = mConfig.nTree * sizeof(blk);

  for (uint64_t d = 0, inWidth = 1; d < mDepth; d++, inWidth *= 2) {
    std::swap(input, output);
    expander->expand(*input, *output, separated, mConfig.nTree*inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);

    gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d, 0}), (uint8_t*)separated.data(), numBytes);
    gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d, 0}), (uint8_t*)separated.data({mConfig.nTree}), numBytes);

    if (d == mDepth-1) {
      gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*)separated.data({mConfig.nTree}), numBytes);
      gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*)separated.data(), numBytes);
      xor_single<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*) delta, sizeof(blk), numBytes);
      xor_single<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*) delta, sizeof(blk), numBytes);
    }
  }
  fullVector = output;
  buffer = input;
  cudaDeviceSynchronize();
  Log::mem(Sender, SeedExp);
}

void SilentOTSender::dual_lpn() {
  cudaSetDevice(mConfig.id);
  Log::mem(Sender, LPN);
  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  fullVector->bit_transpose(mConfig.id*rowsPerGPU, (mConfig.id+1)*rowsPerGPU);
  lpn->encode_dense(*fullVector);
  fullVector->bit_transpose();
  cudaDeviceSynchronize();
  Log::mem(Sender, LPN);
}
