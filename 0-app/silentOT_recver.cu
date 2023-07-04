#include "rand.h"
#include "aes.h"
#include "simplest_ot.h"
#include "silentOT.h"
#include "basic_op.h"
#include <future>

std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

SilentOTRecver::SilentOTRecver(int myid, int logOT, int numTrees, uint64_t *mychoices) :
  SilentOT(myid, logOT, numTrees){

  choices = mychoices;
  expandEvents.resize(depth);
  for (auto &event : expandEvents) {
    cudaEventCreate(&event);
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

void SilentOTRecver::run() {
  Log::start(Recver, BaseOT);
  baseOT();
  Log::end(Recver, BaseOT);

  Log::start(Recver, BufferInit);
  buffer_init();
  Log::end(Recver, BufferInit);

  Log::start(Recver, PprfExpand);
  expand();
  get_choice_vector();
  Log::end(Recver, PprfExpand);
  return;

  Log::start(Recver, MatrixInit);
  QuasiCyclic code(2 * numOT, numOT);
  Log::end(Recver, MatrixInit);

  Log::start(Recver, MatrixMult);
  code.encode(puncVector);
  Log::end(Recver, MatrixMult);
}

void SilentOTRecver::baseOT() {
  std::vector<std::future<GPUBlock>> workers;
   for (int d = 0; d < depth+1; d++) {
    workers.push_back(std::async([d, this]() {
      return SimplestOT(SimplestOT::Recver, d, nTree).recv(rand());
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

void SilentOTRecver::buffer_init() {
  puncVector.resize(2 * numOT * sizeof(OTBlock));

  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aesLeft.init(k0_blk);
  aesRight.init(k1_blk);

  bufferA.resize(2 * numOT * sizeof(OTBlock));
  bufferB.resize(2 * numOT * sizeof(OTBlock));

  leftHash.resize(depth+1);
  rightHash.resize(depth+1);
  leftNodes.resize(numOT * sizeof(OTBlock));
  rightNodes.resize(numOT * sizeof(OTBlock));
}

void SilentOTRecver::get_choice_vector() {
  uint64_t *choices_d;
  choiceVector.resize(2 * numOT / 8);
  cudaMalloc(&choices_d, nTree * sizeof(*choices_d));
  cudaMemcpy(choices_d, choices, nTree * sizeof(*choices_d), cudaMemcpyHostToDevice);
  pathToChoice<<<1, nTree>>>(choiceVector.data_d, depth, numLeaves, choices_d);
  cudaDeviceSynchronize();
}

void SilentOTRecver::expand() {
  std::vector<uint64_t> activeParent(nTree, 0);
  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  GPUBlock *inBuffer, *outBuffer;
  GPUBlock recvSums(nTree * sizeof(OTBlock));
  GPUBlock *tmp;
  uint8_t choice;
  size_t offsetInVec;

  while(!eventsRecorded);
  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
    outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;
    OTBlock *inPtr = (OTBlock*) inBuffer->data_d;
    OTBlock *outPtr = (OTBlock*) outBuffer->data_d;

    uint64_t packedWidth = nTree * width;
    aesLeft.expand_async(outPtr, leftNodes, inPtr, packedWidth, 0, stream[0]);
    aesRight.expand_async(outPtr, rightNodes, inPtr, packedWidth, 1, stream[0]);

    cudaStreamWaitEvent(stream[0], expandEvents.at(d-1));
    cudaStreamWaitEvent(stream[0], expandEvents.at(d-1));

    for (uint64_t t = 0; t < nTree; t++) {
      // copy from left or right hashed sum into vector and hash the vector
      // with choice values obtained from BaseOT to obtain the sum values
      choice = choices[t] >> (d-1) & 1;
      tmp = choice == 0 ? &leftHash.at(d-1) : &rightHash.at(d-1);
      cudaMemcpyAsync((OTBlock*)recvSums.data_d+t, (OTBlock*)tmp->data_d+t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      if (d == depth) {
        tmp = choice == 0 ? &rightHash.at(d) : &leftHash.at(d);
        cudaMemcpyAsync((OTBlock*)recvSums.data_d+t, (OTBlock*)tmp->data_d+t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      }
    }

    choiceHash.at(d-1).xor_async(recvSums, stream[0]);
    if (d == depth)
      choiceHash.at(d).xor_async(recvSums, stream[0]);

    for (uint64_t t = 0; t < nTree; t++) {
      // insert obtained sum into left side or right side
      // and hash to retrieve active node value
      choice = choices[t] >> (d-1) & 1;
      tmp = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width / 2 + activeParent.at(t);
      cudaMemcpyAsync((OTBlock*)tmp->data_d + offsetInVec, (OTBlock*)choiceHash.at(d-1).data_d + t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      if (d == depth) {
        tmp = choice == 0 ? &rightNodes : &leftNodes;
        cudaMemcpyAsync((OTBlock*)tmp->data_d + offsetInVec, (OTBlock*)choiceHash.at(d).data_d + t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      }
    }
    leftNodes.sum_async(nTree, width / 2, stream[0]);
    rightNodes.sum_async(nTree, width / 2, stream[0]);

    // insert active node value obtained from sum into output
    for (uint64_t t = 0; t < nTree; t++) {
      choice = choices[t] >> (d-1) & 1;
      tmp = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width + 2 * activeParent.at(t) + choice;
      cudaMemcpyAsync((OTBlock*)outPtr + offsetInVec, (OTBlock*)tmp->data_d + t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      if (d == depth) {
        tmp = choice == 0 ? &rightNodes : &leftNodes;
        offsetInVec = t * width + 2 * activeParent.at(t) + (1-choice);
        cudaMemcpyAsync((OTBlock*)outPtr + offsetInVec, (OTBlock*)tmp->data_d + t, sizeof(OTBlock), cudaMemcpyDeviceToDevice, stream[0]);
      }
    }
  }
  cudaDeviceSynchronize();
  eventsRecorded = false;
  cudaStreamDestroy(stream[0]);
  // cudaStreamDestroy(stream[1]);
  puncVector = *outBuffer;
}
