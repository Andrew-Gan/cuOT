#include "roles.h"
#include <future>
#include "event_log.h"

std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) :
  SilentOT(config), fullVector(2 * numOT) {
  expandEvents.resize(depth);
  for (auto &event : expandEvents) {
    cudaEventCreate(&event);
  }
  silentOTSenders[config.id] = this;
  while(silentOTRecvers[config.id] == nullptr);
  other = silentOTRecvers[config.id];
}

void SilentOTSender::run() {
  Log::start(Sender, BaseOT);
  base_ot();
  Log::end(Sender, BaseOT);

  Log::start(Sender, SeedExp);
  pprf_expand();
  Log::end(Sender, SeedExp);

  Log::start(Sender, LPN);
  mult_compress();
  Log::end(Sender, LPN);
}

void SilentOTSender::base_ot() {
  std::vector<std::future<std::array<Vec, 2>>> workers;
  for (int d = 0; d <= depth; d++) {
    workers.push_back(std::async([d, this]() {
      switch (mConfig.baseOT) {
        case SimplestOT_t: return SimplestOT(Sender, d, mConfig.nTree).send();
      }
      return std::array<Vec, 2>();
    }));
  }

  for (auto &worker : workers) {
    auto res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
}

void SilentOTSender::pprf_expand() {
  // init hash keys
  uint32_t k0_blk[4] = {3242342};
  uint32_t k1_blk[4] = {8993849};
  Expand *expander;
  switch (mConfig.expander) {
    case AesExpand_t:
      expander = new AesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  }
  Vec separated(2 * numOT);
  Vec leftSum(mConfig.nTree), rightSum(mConfig.nTree);
  // init delta
  blk buff;
  for (int i = 0; i < 4; i++) {
    buff.data[i] = rand();
  }
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &buff, sizeof(*delta), cudaMemcpyHostToDevice);
  // init root
  for (int t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) {
      buff.data[i] = rand();
    }
    fullVector.set(t, buff);
  }

  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    expander->expand(fullVector, separated, mConfig.nTree * inWidth);

    separated.sum(2 * mConfig.nTree, inWidth);

    leftHash.at(d).xor_d(separated, 0);
    rightHash.at(d).xor_d(separated, mConfig.nTree);

    other->leftBuffer.at(d).copy(leftHash.at(d));
    other->rightBuffer.at(d).copy(rightHash.at(d));

    if (d == depth-1) {
      leftHash.at(d+1).xor_d(separated, 0);
      rightHash.at(d+1).xor_d(separated, mConfig.nTree);

      leftHash.at(d+1).xor_scalar(delta);
      rightHash.at(d+1).xor_scalar(delta);

      other->leftBuffer.at(d+1).copy(leftHash.at(d+1));
      other->rightBuffer.at(d+1).copy(rightHash.at(d+1));
    }

    cudaEventRecord(expandEvents.at(d));
  }

  other->eventsRecorded = true;
  cudaDeviceSynchronize();

  delete expander;
}

void SilentOTSender::mult_compress() {
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      QuasiCyclic code(Sender, 2 * numOT, numOT);
      code.encode(fullVector);
    // case ExpandAccumulate:
  }
}
