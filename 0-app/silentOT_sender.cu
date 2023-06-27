#include "rand.h"
#include "aes.h"
#include "simplest_ot.h"
#include "silentOT.h"
#include "basic_op.h"
#include <future>

SilentOTSender::SilentOTSender(int myid, int logOT, int numTrees) :
  SilentOT(myid, logOT, numTrees) {

  silentOTSenders[id] = this;
  while(silentOTRecvers[id] == nullptr);
  other = silentOTRecvers[id];
}

std::pair<GPUBlock, GPUBlock> SilentOTSender::run() {
  EventLog::start(Sender, BaseOT);
  baseOT();
  EventLog::end(Sender, BaseOT);

  EventLog::start(Sender, BufferInit);
  delta.resize(BLK_SIZE);
  EventLog::end(Sender, BufferInit);

  expand();

  GPUBlock fullVectorHashed(numOT * BLK_SIZE);

  // if (numOT < CHUNK_SIDE) {
  //   EventLog::start(Sender, MatrixInit);
  //   randMatrix = init_rand(prng, 2 * numOT, numOT);
  //   EventLog::end(Sender, MatrixInit);
  //   EventLog::start(Sender, MatrixRand);
  //   gen_rand(prng, randMatrix); // transposed
  //   EventLog::end(Sender, MatrixRand);
  //   EventLog::start(Sender, MatrixMult);
  //   compress(fullVectorHashed, randMatrix, fullVector, 0);
  //   EventLog::end(Sender, MatrixMult);
  // }
  // else {
  //   EventLog::start(Sender, MatrixInit);
  //   randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
  //   EventLog::end(Sender, MatrixInit);
  //   for (uint64_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
  //     for (uint64_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
  //       EventLog::start(Sender, MatrixRand);
  //       gen_rand(prng, randMatrix);
  //       EventLog::end(Sender, MatrixRand);
  //       EventLog::start(Sender, MatrixMult);
  //       compress(fullVectorHashed, randMatrix, fullVector, chunkC);
  //       EventLog::end(Sender, MatrixMult);
  //     }
  //   }
  // }
  // del_rand(prng, randMatrix);
  return {fullVectorHashed, delta};
}

void SilentOTSender::baseOT() {
  std::vector<std::future<std::array<std::vector<GPUBlock>, 2>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Sender, t).send(depth+1);
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
}

void SilentOTSender::expand() {
  EventLog::start(Sender, BufferInit);
  OTBlock root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  delta.clear();
  delta.set(123456);

  GPUBlock bufferA(2 * numOT * BLK_SIZE);
  GPUBlock bufferB(2 * numOT * BLK_SIZE);
  std::vector<GPUBlock> leftNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  std::vector<GPUBlock> rightNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  Aes aesLeft(k0_blk);
  Aes aesRight(k1_blk);

  for (int t = 0; t < nTree; t++) {
    bufferA.set((uint8_t*) root.data, BLK_SIZE, t * numLeaves * BLK_SIZE);
  }
  std::vector<cudaStream_t> streams(nTree);
  for (cudaStream_t &s : streams) {
    cudaStreamCreate(&s);
  }
  EventLog::end(Sender, BufferInit);

  EventLog::start(Sender, PprfExpand);
  GPUBlock *inBuffer, *outBuffer;
  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (uint64_t t = 0; t < nTree; t++) {
      cudaStream_t &stream = streams.at(t);

      inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
      outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;

      OTBlock *inPtr = (OTBlock*)inBuffer->data_d + t * numLeaves;
      OTBlock *outPtr = (OTBlock*)outBuffer->data_d + t * numLeaves;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0, stream);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1, stream);

      leftNodes.at(t).sum_async(BLK_SIZE * width / 2, stream);
      rightNodes.at(t).sum_async(BLK_SIZE * width / 2, stream);

      leftHash.at(t).at(d-1).xor_async(leftNodes.at(t), stream);
      rightHash.at(t).at(d-1).xor_async(rightNodes.at(t), stream);

      other->leftHash.at(t).at(d-1).copy_async(leftHash.at(t).at(d-1), stream);
      other->rightHash.at(t).at(d-1).copy_async(rightHash.at(t).at(d-1), stream);

      if (d == depth) {
        leftHash.at(t).at(d).xor_async(leftNodes.at(t), stream);
        rightHash.at(t).at(d).xor_async(rightNodes.at(t), stream);

        leftHash.at(t).at(d).xor_async(delta, stream);
        rightHash.at(t).at(d).xor_async(delta, stream);

        other->leftHash.at(t).at(d).copy_async(leftHash.at(t).at(d), stream);
        other->rightHash.at(t).at(d).copy_async(rightHash.at(t).at(d), stream);
      }

      cudaEventRecord(other->expandEvents.at(t).at(d-1), stream);
    }
  }
  other->eventsRecorded = true;
  cudaDeviceSynchronize();
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
  fullVector = *outBuffer;
  EventLog::end(Sender, PprfExpand);
}
