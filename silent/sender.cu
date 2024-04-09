#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentConfig config) : SilentOT(config) {
  blk seed_h, delta_h;
  mRole = Sender;
  mDev = mConfig.id;
  cudaSetDevice(mDev);
  silentOTSenders[mConfig.id] = this;
  for (int i = 0; i < 4; i++)
    delta_h.data[i] = rand();

  fullVector = new Mat({2 * numOT, 1});
  buffer = new Mat({2 * numOT, 1});
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &delta_h, sizeof(*delta), cudaMemcpyHostToDevice);
  for (uint64_t t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) seed_h.data[i] = rand();
    fullVector->set(seed_h, {t, 0});
  }
  separated.resize({numOT});
  switch (mConfig.pprf) {
    case AesExpand_t:
      expander = new AesExpand(mConfig.leftKey, mConfig.rightKey);
  }

  uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
  b64.resize({rowsPerGPU, 2 * numOT / BLOCK_BITS});
  b64.clear();
  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Sender, 2 * numOT, numOT, BLOCK_BITS / NGPU);
  }
}

SilentOTSender::~SilentOTSender() {
  cudaSetDevice(mDev);
  delete fullVector;
  delete buffer;
  delete expander;
  delete lpn;
  cudaFree(delta);
  silentOTSenders[mConfig.id] = nullptr;
}

void SilentOTSender::base_ot() {
  cudaSetDevice(mDev);
  std::vector<std::future<std::array<Mat, 2>>> workers;
  for (int d = 0; d < depth; d++) {
    workers.push_back(std::async([d, this]() {
      cudaSetDevice(mConfig.id);
      switch (mConfig.baseOT) {
        case SimplestOT_t:
          return SimplestOT(Sender, this->mConfig.id*this->depth+d, mConfig.nTree).send();
      }
      return std::array<Mat, 2>();
    }));
  }
  for (auto &worker : workers) {
    std::array<Mat, 2> res = worker.get();
    m0.push_back(res[0]);
    m1.push_back(res[1]);
  }
  m0.push_back(m0.back());
  m1.push_back(m1.back());
}

void SilentOTSender::seed_expand() {
  cudaSetDevice(mDev);
  Log::mem(Sender, SeedExp);
  Mat *input = buffer;
  Mat *output = fullVector;
  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    std::swap(input, output);
    expander->expand(*input, *output, separated, mConfig.nTree*inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);
    m0.at(d).xor_d(separated, 0);
    m1.at(d).xor_d(separated, mConfig.nTree);

    if (d == depth-1) {
      m0.at(d+1).xor_d(separated, mConfig.nTree);
      m1.at(d+1).xor_d(separated, 0);
      m0.at(d+1).xor_scalar(delta);
      m1.at(d+1).xor_scalar(delta);
    }
  }
  fullVector = output;
  buffer = input;
  cudaDeviceSynchronize();
  Log::mem(Sender, SeedExp);
}

void SilentOTSender::dual_lpn() {
  cudaSetDevice(mDev);
  Log::mem(Sender, LPN);
  uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
  fullVector->bit_transpose();
  Span b64(*fullVector, {mConfig.id*rowsPerGPU, 0}, {(mConfig.id+1)*rowsPerGPU, 0});
  lpn->encode_dense(b64);
  fullVector->resize({fullVector->dim(0), numOT / BLOCK_BITS});
  cudaDeviceSynchronize();
  Log::mem(Sender, LPN);
}
