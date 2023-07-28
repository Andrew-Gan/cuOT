#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) :
  SilentOT(config), fullVector(2 * numOT) {
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

  return;

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
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  Expander *expandLeft, *expandRight;
  switch (mConfig.expander) {
    case AesHash_t:
      expandLeft = new AesHash(k0_blk);
      expandRight = new AesHash(k1_blk);
  }

  // init buffers
  GPUvector<OTblock> bufferA(2 * numOT), bufferB(2 * numOT);
  GPUvector<OTblock> leftNodes(numOT), rightNodes(numOT);
  GPUvector<OTblock> leftSum(mConfig.nTree), rightSum(mConfig.nTree);

  // init root
  OTblock buff;
  for (int i = 0; i < 4; i++) {
    buff.data[i] = rand();
  }
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &buff, sizeof(*delta), cudaMemcpyHostToDevice);
  for (int t = 0; t < mConfig.nTree; t++) {
    for (int i = 0; i < 4; i++) {
      buff.data[i] = i;
    }
    bufferA.set(t, buff);
  }

  cudaStream_t stream[4];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);
  cudaStreamCreate(&stream[3]);
  GPUvector<OTblock> *inBuffer, *outBuffer;

  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
    outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;
    OTblock *inPtr = inBuffer->data();
    OTblock *outPtr = outBuffer->data();

    uint64_t packedWidth = mConfig.nTree * width;
    expandLeft->expand_async(outPtr, leftNodes, inPtr, packedWidth, 0, stream[0]);
    expandRight->expand_async(outPtr, rightNodes, inPtr, packedWidth, 1, stream[1]);

    leftNodes.sum_async(mConfig.nTree, width / 2, stream[0]);
    rightNodes.sum_async(mConfig.nTree, width / 2, stream[1]);

    cudaMemcpyAsync(leftSum.data(), leftNodes.data(), mConfig.nTree * sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[0]);
    cudaMemcpyAsync(rightSum.data(), rightNodes.data(), mConfig.nTree * sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[1]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    leftHash.at(d-1).xor_async(leftSum, stream[2]);
    rightHash.at(d-1).xor_async(rightSum, stream[3]);

    if (d == depth) {
      leftHash.at(d).xor_async(leftSum,stream[2]);
      rightHash.at(d).xor_async(rightSum, stream[3]);
    }

    other->leftBuffer.at(d-1).copy_async(leftHash.at(d-1), stream[2]);
    other->rightBuffer.at(d-1).copy_async(rightHash.at(d-1), stream[3]);

    if (d == depth) {
      leftHash.at(d).xor_one_to_many_async(delta, stream[2]);
      rightHash.at(d).xor_one_to_many_async(delta, stream[3]);

      other->leftBuffer.at(d).copy_async(leftHash.at(d), stream[2]);
      other->rightBuffer.at(d).copy_async(rightHash.at(d), stream[3]);
    }

    cudaEventRecord(other->expandEvents.at(d-1), stream[2]);
    cudaEventRecord(other->expandEvents.at(d-1), stream[3]);
  }

  other->eventsRecorded = true;
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cudaStreamDestroy(stream[2]);
  cudaStreamDestroy(stream[3]);

  fullVector = *outBuffer;

  delete expandLeft;
  delete expandRight;
}

void SilentOTSender::mult_compress() {
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      QuasiCyclic code(Sender, 2 * numOT, numOT);
      code.encode(fullVector);
    // case ExpandAccumulate:
  }
}
