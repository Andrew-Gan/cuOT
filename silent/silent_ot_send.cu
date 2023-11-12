#include "silent_ot.h"
#include <future>

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

  Log::start(Sender, Expand);
  pprf_expand();
  Log::end(Sender, Expand);

  Log::start(Sender, Compress);
  mult_compress();
  Log::end(Sender, Compress);
}

void SilentOTSender::base_ot() {
  std::vector<std::future<std::array<vec, 2>>> workers;
  for (int d = 0; d <= depth; d++) {
    workers.push_back(std::async([d, this]() {
      switch (mConfig.baseOT) {
        case SimplestOT_t: return SimplestOT(Sender, d, mConfig.nTree).send();
      }
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
  Expander *expander;
  switch (mConfig.expander) {
    case AesHash_t:
      expander = new AesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  }
  // init buffers
  vec interleaved(2 * numOT);
  vec separated(2 * numOT);
  vec leftSum(mConfig.nTree), rightSum(mConfig.nTree);
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
    interleaved.set(t, buff);
  }

  vec *inBuffer, *outBuffer;

  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &interleaved : &fullVector;
    outBuffer = (d % 2 == 1) ? &fullVector : &interleaved;

    uint64_t packedWidth = mConfig.nTree * width;
    expander->expand(*outBuffer, separated, *inBuffer, packedWidth);

    separated.sum(2 * mConfig.nTree, width / 2);

    leftHash.at(d-1).xor_d(separated, 0);
    rightHash.at(d-1).xor_d(separated, mConfig.nTree);

    other->leftBuffer.at(d-1).copy(leftHash.at(d-1));
    other->rightBuffer.at(d-1).copy(rightHash.at(d-1));

    if (d == depth) {
      leftHash.at(d).xor_d(separated, 0);
      rightHash.at(d).xor_d(separated, mConfig.nTree);

      leftHash.at(d).xor_scalar(delta);
      rightHash.at(d).xor_scalar(delta);

      other->leftBuffer.at(d).copy(leftHash.at(d));
      other->rightBuffer.at(d).copy(rightHash.at(d));
    }

    cudaEventRecord(expandEvents.at(d-1));
  }

  other->eventsRecorded = true;
  cudaDeviceSynchronize();

  if (outBuffer != &fullVector)
    fullVector = *outBuffer;

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
