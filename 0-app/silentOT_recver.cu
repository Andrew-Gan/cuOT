#include "rand.h"
#include "aes.h"
#include "simplest_ot.h"
#include "silentOT.h"
#include "basic_op.h"
#include <future>

SilentOTRecver::SilentOTRecver(int myid, int logOT, int numTrees, uint64_t *mychoices) :
  SilentOT(myid, logOT, numTrees){

  choices = mychoices;
  expandEvents = std::vector<std::vector<cudaEvent_t>>(nTree, std::vector<cudaEvent_t>(depth));
  for (auto &depths : expandEvents) {
    for (auto &event : depths) {
      cudaEventCreate(&event);
    }
  }
  silentOTRecvers[id] = this;
  while(silentOTSenders[id] == nullptr);
  other = silentOTSenders[id];
}

__global__
void pathToChoice(uint8_t *choiceVec, uint64_t depth, uint64_t numLeaves, uint64_t *choices) {
  uint64_t treeStartIndex = threadIdx.x * numLeaves;
  uint64_t path = choices[threadIdx.x];
  uint64_t puncIndex = 0;
  for (int d = 0; d < depth; d++) {
    puncIndex *= 2;
    if (path & (1 << d)) puncIndex += 1;
  }
  puncIndex += treeStartIndex;
  choiceVec[puncIndex / 8] |= 1 << (puncIndex % 8);
}

std::pair<GPUBlock, GPUBlock> SilentOTRecver::run() {
  EventLog::start(Recver, BaseOT);
  baseOT();
  EventLog::end(Recver, BaseOT);

  EventLog::start(Recver, BufferInit);
  puncVector.resize(2 * numOT * BLK_SIZE);
  EventLog::end(Recver, BufferInit);

  expand();

  GPUBlock choiceVector(2 * numOT / 8);
  uint64_t *choices_d;
  cudaMalloc(&choices_d, nTree * sizeof(*choices_d));
  cudaMemcpy(choices_d, choices, nTree * sizeof(*choices_d), cudaMemcpyHostToDevice);
  pathToChoice<<<1, nTree>>>(choiceVector.data_d, depth, numLeaves, choices_d);
  cudaDeviceSynchronize();

  GPUBlock puncVectorHashed(numOT * BLK_SIZE);
  GPUBlock choiceVectorHashed(numOT * BLK_SIZE);

  // SparseVector choiceVector;

  // if (numOT < CHUNK_SIDE) {
  //   EventLog::start(Recver, MatrixInit);
  //   randMatrix = init_rand(prng, 2 * numOT, numOT);
  //   EventLog::end(Recver, MatrixInit);
  //   EventLog::start(Recver, MatrixRand);
  //   gen_rand(prng, randMatrix); // transposed
  //   EventLog::end(Recver, MatrixRand);
  //   EventLog::start(Recver, MatrixMult);
  //   compress(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, 0, 0);
  //   EventLog::end(Recver, MatrixMult);
  // }
  // else {
  //   EventLog::start(Recver, MatrixInit);
  //   randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
  //   EventLog::end(Recver, MatrixInit);
  //   for (uint64_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
  //     for (uint64_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
  //       EventLog::start(Recver, MatrixRand);
  //       gen_rand(prng, randMatrix);
  //       EventLog::end(Recver, MatrixRand);
  //       EventLog::start(Recver, MatrixMult);
  //       compress(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, chunkR, chunkC);
  //       EventLog::end(Recver, MatrixMult);
  //     }
  //   }
  // }
  // del_rand(prng, randMatrix);
  return {puncVectorHashed, choiceVectorHashed};
}

void SilentOTRecver::baseOT() {
  std::vector<std::future<std::vector<GPUBlock>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Recver, t).recv(depth+1, rand());
    }));
  }
  leftHash.resize(nTree);
  rightHash.resize(nTree);
  for (int i = 0; i < nTree; i++) {
    leftHash.at(i).resize(depth+1);
    rightHash.at(i).resize(depth+1);
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

void SilentOTRecver::expand() {
  EventLog::start(Recver, BufferInit);
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  GPUBlock bufferA(2 * numOT * BLK_SIZE);
  GPUBlock bufferB(2 * numOT * BLK_SIZE);
  std::vector<GPUBlock> leftNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  std::vector<GPUBlock> rightNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  Aes aesLeft(k0_blk);
  Aes aesRight(k1_blk);
  std::vector<uint64_t> puncture(nTree, 0);

  std::vector<cudaStream_t> streams(nTree);
  for (cudaStream_t &s : streams) {
    cudaStreamCreate(&s);
  }
  EventLog::end(Recver, BufferInit);

  while(!eventsRecorded);
  EventLog::start(Recver, PprfExpand);
  GPUBlock *inBuffer, *outBuffer;
  auto &sum = choiceHash; // alias
  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    for (uint64_t t = 0; t < nTree; t++) {
      cudaStream_t &stream = streams.at(t);

      inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
      outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;

      OTBlock *inPtr = (OTBlock*)inBuffer->data_d + t * numLeaves;
      OTBlock *outPtr = (OTBlock*)outBuffer->data_d + t * numLeaves;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0, stream);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1, stream);

      cudaStreamWaitEvent(stream, expandEvents.at(t).at(d-1));

      // once left sum^hash and right sum^hash ready, unhash to obtain sum
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      if (choice == 0)
        sum.at(t).at(d-1).xor_async(leftHash.at(t).at(d-1), stream);
      else
        sum.at(t).at(d-1).xor_async(rightHash.at(t).at(d-1), stream);

      if (d == depth) {
        if (choice == 0)
          sum.at(t).at(d).xor_async(rightHash.at(t).at(d), stream);
        else
          sum.at(t).at(d).xor_async(leftHash.at(t).at(d), stream);
      }

      // insert obtained sum into layer
      choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      OTBlock *sideCasted = (OTBlock*) side->data_d;
      int recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpyAsync(&sideCasted[recvNodeId / 2], sum.at(t).at(d-1).data_d, BLK_SIZE, cudaMemcpyDeviceToDevice, stream);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        sideCasted = (OTBlock*) xorSide->data_d;
        uint64_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpyAsync(&sideCasted[deltaNodeId / 2], sum.at(t).at(d).data_d, BLK_SIZE, cudaMemcpyDeviceToDevice, stream);
      }

      // conduct sum/xor in parallel
      choice = (choices[t] & (1 << d-1)) >> d-1;
      side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      side->sum_async(BLK_SIZE * width / 2, stream);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        xorSide->sum_async(BLK_SIZE * width / 2, stream);
      }

      // insert active node obtained from sum into output
      choice = (choices[t] & (1 << d-1)) >> d-1;
      side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      OTBlock *oCasted = (OTBlock*) puncVector.data_d + t * numLeaves;
      recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpyAsync(&oCasted[recvNodeId], side->data_d, BLK_SIZE, cudaMemcpyDeviceToDevice, stream);

      if(d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        uint64_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpyAsync(&oCasted[deltaNodeId], xorSide->data_d, BLK_SIZE, cudaMemcpyDeviceToDevice, stream);
      }
    }
  }
  cudaDeviceSynchronize();
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
  puncVector = *outBuffer;
  EventLog::end(Recver, PprfExpand);
}
