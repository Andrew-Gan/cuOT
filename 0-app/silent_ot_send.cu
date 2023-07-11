#include "simplest_ot.h"
#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;

SilentOTSender::SilentOTSender(int myid, int logOT, int numTrees) :
  SilentOT(myid, logOT, numTrees) {

  silentOTSenders[id] = this;
  while(silentOTRecvers[id] == nullptr);
  other = silentOTRecvers[id];
}

void SilentOTSender::run() {
  Log::start(Sender, BaseOT);
  baseOT();
  Log::end(Sender, BaseOT);

  Log::start(Sender, BufferInit);
  buffer_init();
  Log::end(Sender, BufferInit);

  Log::start(Sender, PprfExpand);
  expand();
  Log::end(Sender, PprfExpand);

  Log::start(Sender, MatrixInit);
  QuasiCyclic code(2 * numOT, numOT);
  Log::end(Sender, MatrixInit);

  Log::start(Sender, MatrixMult);
  code.encode(fullVector);
  Log::end(Sender, MatrixMult);
}

void SilentOTSender::baseOT() {
  std::vector<std::future<std::array<GPUvector<OTblock>, 2>>> workers;
  for (int d = 0; d < depth+1; d++) {
    workers.push_back(std::async([d, this]() {
      return SimplestOT(SimplestOT::Sender, d, nTree).send();
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
}

void SilentOTSender::buffer_init() {
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aesLeft.init(k0_blk);
  aesRight.init(k1_blk);

  cudaMalloc(&delta, sizeof(*delta));
  // delta.set(123456);

  bufferA.resize(2 * numOT);
  bufferB.resize(2 * numOT);
  leftNodes.resize(numOT);
  rightNodes.resize(numOT);

  OTblock root;
  for (int t = 0; t < nTree; t++) {
    root.data[0] = rand();
    root.data[1] = rand();
    bufferA.set(t, root);
  }
}

void SilentOTSender::expand() {
  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  GPUvector<OTblock> *inBuffer, *outBuffer;

  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
    outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;
    OTblock *inPtr = inBuffer->data();
    OTblock *outPtr = outBuffer->data();

    uint64_t packedWidth = nTree * width;
    aesLeft.expand_async(outPtr, leftNodes, inPtr, packedWidth, 0, stream[0]);
    aesRight.expand_async(outPtr, rightNodes, inPtr, packedWidth, 1, stream[1]);

    leftNodes.sum_async(nTree, width / 2, stream[0]);
    rightNodes.sum_async(nTree, width / 2, stream[1]);

    leftHash.at(d-1).xor_async(leftNodes, stream[0]);
    rightHash.at(d-1).xor_async(rightNodes, stream[1]);

    other->leftHash.at(d-1).copy_async(leftHash.at(d-1), stream[0]);
    other->rightHash.at(d-1).copy_async(rightHash.at(d-1), stream[1]);

    if (d == depth) {
      leftHash.at(d).xor_async(leftNodes, stream[0]);
      rightHash.at(d).xor_async(rightNodes, stream[1]);

      leftHash.at(d).xor_async(delta, stream[0]);
      rightHash.at(d).xor_async(delta, stream[1]);

      other->leftHash.at(d).copy_async(leftHash.at(d), stream[0]);
      other->rightHash.at(d).copy_async(rightHash.at(d), stream[1]);
    }

    cudaEventRecord(other->expandEvents.at(d-1), stream[0]);
    cudaEventRecord(other->expandEvents.at(d-1), stream[1]);
  }
  other->eventsRecorded = true;
  cudaDeviceSynchronize();
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  fullVector = *outBuffer;
}
