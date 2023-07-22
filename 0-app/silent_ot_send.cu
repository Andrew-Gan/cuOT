#include "simplest_ot.h"
#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;

SilentOTSender::SilentOTSender(int myid, int logOT, int numTrees) :
  SilentOT(myid, logOT, numTrees) {

  buffer_init();
  silentOTSenders[id] = this;
  while(silentOTRecvers[id] == nullptr);
  other = silentOTRecvers[id];
}

void SilentOTSender::run() {
  Log::start(Sender, BaseOT);
  base_ot();
  Log::end(Sender, BaseOT);

  Log::start(Sender, Expand);
  pprf_expand();
  Log::end(Sender, Expand);

  Log::start(Sender, Compress);
  QuasiCyclic code(Sender, 2 * numOT, numOT);
  code.encode(fullVector);
  Log::end(Sender, Compress);
}

void SilentOTSender::base_ot() {
  std::vector<std::future<std::array<GPUvector<OTblock>, 2>>> workers;
  for (int d = 0; d <= depth; d++) {
    workers.push_back(std::async([d, this]() {
      return SimplestOT(Sender, d, nTree).send();
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

  OTblock buff;

  for (int i = 0; i < 4; i++) {
    buff.data[i] = rand();
  }
  cudaMalloc(&delta, sizeof(*delta));
  cudaMemcpy(delta, &buff, sizeof(*delta), cudaMemcpyHostToDevice);

  bufferA.resize(2 * numOT);
  bufferB.resize(2 * numOT);
  leftNodes.resize(numOT);
  rightNodes.resize(numOT);

  leftSum.resize(nTree);
  rightSum.resize(nTree);

  for (int t = 0; t < nTree; t++) {
    for (int i = 0; i < 4; i++) {
      buff.data[i] = i;
    }
    bufferA.set(t, buff);
  }
}

void SilentOTSender::pprf_expand() {
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

    uint64_t packedWidth = nTree * width;
    aesLeft.expand_async(outPtr, leftNodes, inPtr, packedWidth, 0, stream[0]);
    aesRight.expand_async(outPtr, rightNodes, inPtr, packedWidth, 1, stream[1]);

    leftNodes.sum_async(nTree, width / 2, stream[0]);
    rightNodes.sum_async(nTree, width / 2, stream[1]);

    cudaMemcpyAsync(leftSum.data(), leftNodes.data(), nTree * sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[0]);
    cudaMemcpyAsync(rightSum.data(), rightNodes.data(), nTree * sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[1]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    leftHash.at(d-1).xor_async(leftSum, stream[2]);
    rightHash.at(d-1).xor_async(rightSum, stream[3]);

    if (d == depth) {
      leftHash.at(d).xor_async(leftSum,stream[2]);
      rightHash.at(d).xor_async(rightSum, stream[3]);
    }

    other->leftHash.at(d-1).copy_async(leftHash.at(d-1), stream[2]);
    other->rightHash.at(d-1).copy_async(rightHash.at(d-1), stream[3]);

    if (d == depth) {
      leftHash.at(d).xor_one_to_many_async(delta, stream[2]);
      rightHash.at(d).xor_one_to_many_async(delta, stream[3]);

      other->leftHash.at(d).copy_async(leftHash.at(d), stream[2]);
      other->rightHash.at(d).copy_async(rightHash.at(d), stream[3]);
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
}
