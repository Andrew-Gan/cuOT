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
  std::vector<std::future<std::array<GPUvector<OTblock>, 2>>> workers;
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
  GPUvector<OTblock> interleaved(2 * numOT);
  GPUvector<OTblock> separated(2 * numOT);
  GPUvector<OTblock> leftSum(mConfig.nTree), rightSum(mConfig.nTree);
  // init delta
  OTblock buff;
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

  cudaStream_t s;
  cudaStreamCreate(&s);
  GPUvector<OTblock> *inBuffer, *outBuffer;

  // struct timespec timePoint[26];

  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &interleaved : &fullVector;
    outBuffer = (d % 2 == 1) ? &fullVector : &interleaved;

    // clock_gettime(CLOCK_MONOTONIC, &timePoint[d-1]);

    uint64_t packedWidth = mConfig.nTree * width;
    expander->expand_async(*outBuffer, separated, *inBuffer, packedWidth, s);

    // cudaDeviceSynchronize();

    // clock_gettime(CLOCK_MONOTONIC, &timePoint[d]);

    separated.sum_async(2 * mConfig.nTree, width / 2, s);

    leftHash.at(d-1).xor_async(separated, 0, s);
    rightHash.at(d-1).xor_async(separated, mConfig.nTree, s);

    other->leftBuffer.at(d-1).copy_async(leftHash.at(d-1), s);
    other->rightBuffer.at(d-1).copy_async(rightHash.at(d-1), s);

    if (d == depth) {
      leftHash.at(d).xor_async(separated, 0, s);
      rightHash.at(d).xor_async(separated, mConfig.nTree, s);

      leftHash.at(d).xor_one_to_many_async(delta, s);
      rightHash.at(d).xor_one_to_many_async(delta, s);

      other->leftBuffer.at(d).copy_async(leftHash.at(d), s);
      other->rightBuffer.at(d).copy_async(rightHash.at(d), s);
    }

    cudaEventRecord(expandEvents.at(d-1), s);
  }

  // for (int i = 0; i < depth; i++) {
  //   float elapsed = (timePoint[i+1].tv_sec - timePoint[i].tv_sec) * 1000;
  //   elapsed += (timePoint[i+1].tv_nsec - timePoint[i].tv_nsec) / 1000000.0;
  //   printf("printing into layer %d: %f\n", i, elapsed);
  // }

  other->eventsRecorded = true;
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

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
