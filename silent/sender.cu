#include "silent_ot.h"
#include <future>

#include <cryptoTools/Crypto/RandomOracle.h>
#include "logger.h"
#include "gpu_tests.h"
#include "gpu_ops.h"
#include "gpu_define.h"

blk* SOTSender::m0_h = nullptr;
blk* SOTSender::m1_h = nullptr;
std::array<std::atomic<SOTSender*>, 16> SOTSenders;

SOTSender::SOTSender(SilentConfig config) : SOT(config) {
  mRole = Sender;
  mGPU = mConfig.id;
  blk seed_h, delta_h;
  cudaSetDevice(mGPU);
  SOTSenders[mConfig.id] = this;
  for (int i = 0; i < 4; i++)
    delta_h.data_32[i] = rand();
  
  m0.resize({mDepth+1, mConfig.nTree});
  m1.resize({mDepth+1, mConfig.nTree});
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &delta_h, sizeof(*delta), cudaMemcpyHostToDevice);

#ifdef USE_COALESCED_TREE_EXPANSION
  fullVector = new Mat({numOT, 1});
  buffer = new Mat(fullVector->dims());
  for (uint64_t t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) seed_h.data_32[i] = rand();
    fullVector->set(seed_h, {t, 0});
  }
  sep = new Mat({numOT});
#else
  fullVector = new Mat[mConfig.nTree];
  buffer = new Mat[mConfig.nTree];
  sep = new Mat[mConfig.nTree];
  for (int t = 0; t < mConfig.nTree; t++) {
    fullVector[t].resize({numOT / mConfig.nTree, 1});
    buffer[t].resize({numOT / mConfig.nTree, 1});
    sep[t].resize({numOT / mConfig.nTree});
  }
  for (uint64_t t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) seed_h.data_32[i] = rand();
    fullVector[t].set(seed_h, {0, 0});
  }
#endif // USE_COALESCED_TREE_EXPANSION

  switch (mConfig.pprf) {
    case Aes_t:
      expander = new Aes(mConfig.leftKey, mConfig.rightKey);
  }
  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Sender, 2 * numOT, numOT, BLOCK_BITS / mConfig.gpuPerParty);
  }

  if (mConfig.id == 0) {
    SOTSender::m0_h = new blk[(mDepth+1) * mConfig.nTree];
    SOTSender::m1_h = new blk[(mDepth+1) * mConfig.nTree];
  }

  CHECK_CUDA("SOTSender::SOTSender")
}

SOTSender::~SOTSender() {
  cudaSetDevice(mGPU);

#ifdef USE_COALESCED_TREE_EXPANSION
  delete fullVector;
  delete buffer;
  delete sep;
#else
  delete[] fullVector;
  delete[] buffer;
  delete[] sep;
#endif // USE_COALESCED_TREE_EXPANSION

  delete expander;
  delete lpn;
  if (mConfig.id == 0) {
    delete[] SOTSender::m0_h;
    delete[] SOTSender::m1_h;
  }
  cudaFree(delta);
  SOTSenders[mConfig.id] = nullptr;
}

void SOTSender::base_ot() {
  cudaSetDevice(mGPU);
  std::vector<std::future<void>> workers;
  for (uint64_t d = 0; d < mDepth; d++) {
    workers.push_back(std::async([d, this](){
      SimplestOT bOT(Sender, d, mConfig.nTree);
      bOT.send(SOTSender::m0_h+d*mConfig.nTree, SOTSender::m1_h+d*mConfig.nTree);
    }));
  }
  for (auto &t : workers) {
    t.get();
  }
}

void SOTSender::seed_exp() {
  cudaSetDevice(mGPU);

  cudaMemcpy(m0.data(), SOTSender::m0_h, m0.size_bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(m1.data(), SOTSender::m1_h, m1.size_bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(m0.data({mDepth, 0}), m0.data({mDepth-1, 0}), m0.dim(1)*sizeof(blk), cudaMemcpyDeviceToDevice);
  cudaMemcpy(m1.data({mDepth, 0}), m1.data({mDepth-1, 0}), m1.dim(1)*sizeof(blk), cudaMemcpyDeviceToDevice);

  Mat *input = buffer;
  Mat *output = fullVector;
  uint64_t numBytes = mConfig.nTree * sizeof(blk);

  for (uint64_t d = 0, inWidth = 1; d < mDepth; d++, inWidth *= 2) {
    std::swap(input, output);
#ifdef USE_COALESCED_TREE_EXPANSION
    expander->expand(*input, *output, *sep, mConfig.nTree*inWidth);
    sep->sum(2 * mConfig.nTree, inWidth);

    gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d, 0}), (uint8_t*)sep->data(), numBytes);
    gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d, 0}), (uint8_t*)sep->data({mConfig.nTree}), numBytes);
    if (d == mDepth-1) {
      gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*)sep->data({mConfig.nTree}), numBytes);
      gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*)sep->data(), numBytes);
      xor_single<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*) delta, sizeof(blk), numBytes);
      xor_single<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*) delta, sizeof(blk), numBytes);
    }
#else
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      expander->expand(input[t], output[t], sep[t], inWidth);
      uint64_t block = std::min(1024UL, 2 * inWidth);
      uint64_t grid = (2 * inWidth + block - 1) / block;
      separator<<<grid, block>>>(sep[t].data(), output[t].data());
      sep[t].sum(2, inWidth);

      gpu_xor<<<1, sizeof(blk)>>>((uint8_t*)m0.data({d, t}), (uint8_t*)sep[t].data(), numBytes);
      gpu_xor<<<1, sizeof(blk)>>>((uint8_t*)m1.data({d, t}), (uint8_t*)sep[t].data({1}), numBytes);
      if (d == mDepth-1) {
        gpu_xor<<<1, sizeof(blk)>>>((uint8_t*)m0.data({d+1, t}), (uint8_t*)sep[t].data({1}), numBytes);
        gpu_xor<<<1, sizeof(blk)>>>((uint8_t*)m1.data({d+1, t}), (uint8_t*)sep[t].data(), numBytes);
        xor_single<<<1, sizeof(blk)>>>((uint8_t*)m0.data({d+1, t}), (uint8_t*) delta, sizeof(blk), numBytes);
        xor_single<<<1, sizeof(blk)>>>((uint8_t*)m1.data({d+1, t}), (uint8_t*) delta, sizeof(blk), numBytes);
      }
    }
#endif // USE_COALESCED_TREE_EXPANSION
  }

  fullVector = output;
  buffer = input;
  CHECK_CUDA("SOTSender::seed_exp")
}

void SOTSender::dual_lpn() {
  cudaSetDevice(mGPU);
  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  fullVector->bit_transpose(mConfig.id*rowsPerGPU, (mConfig.id+1)*rowsPerGPU);
  lpn->encode_dense(*fullVector);
  fullVector->bit_transpose();
  CHECK_CUDA("SOTSender::dual_lpn")
}
