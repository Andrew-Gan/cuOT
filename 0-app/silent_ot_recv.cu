#include "aes.h"
#include "simplest_ot.h"
#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

SilentOTRecver::SilentOTRecver(int myid, int logOT, int numTrees, uint64_t *mychoices) :
  SilentOT(myid, logOT, numTrees){

  choices = mychoices;
  expandEvents.resize(depth);
  for (auto &event : expandEvents) {
    cudaEventCreate(&event);
  }
  buffer_init();
  silentOTRecvers[id] = this;
  while(silentOTSenders[id] == nullptr);
  other = silentOTSenders[id];
}

void SilentOTRecver::run() {
  Log::start(Recver, BaseOT);
  base_ot();
  Log::end(Recver, BaseOT);

  Log::start(Recver, Expand);
  pprf_expand();
  get_choice_vector();
  Log::end(Recver, Expand);

  return;

  Log::start(Recver, Compress);
  QuasiCyclic code(Recver, 2 * numOT, numOT);
  code.encode(puncVector);
  code.encode(choiceVector);
  Log::end(Recver, Compress);
}

void SilentOTRecver::base_ot() {
  std::vector<std::future<GPUvector<OTblock>>> workers;
  for (int d = 0; d <= depth; d++) {
    workers.push_back(std::async([d, this]() {
      return SimplestOT(Recver, d, nTree).recv(choices[d]);
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

void SilentOTRecver::buffer_init() {
  puncVector.resize(2 * numOT);

  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aesLeft.init(k0_blk);
  aesRight.init(k1_blk);

  bufferA.resize(2 * numOT);
  bufferB.resize(2 * numOT);

  leftHash.resize(depth+1);
  rightHash.resize(depth+1);
  leftNodes.resize(numOT);
  rightNodes.resize(numOT);
}

__global__
void pathToChoice(OTblock *choiceVec, uint64_t depth, uint64_t numLeaves, uint64_t *choices) {
  uint64_t treeStartIndex = threadIdx.x * numLeaves;
  uint64_t puncIndex = 0;
  uint8_t path = 0;

  for (int d = 0; d < depth; d++) {
    puncIndex *= 2;
    path = (choices[d] >> threadIdx.x) & 0b1;
    puncIndex += (1-path);
  }
  puncIndex += treeStartIndex;
  for (int i = 0; i < 4; i++) {
    choiceVec[puncIndex].data[i] = 0xffffffff;
  }
}

void SilentOTRecver::get_choice_vector() {
  uint64_t *choices_d;
  choiceVector.resize(2 * numOT);
  choiceVector.clear();
  cudaMalloc(&choices_d, depth * sizeof(*choices_d));
  cudaMemcpy(choices_d, choices, depth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  pathToChoice<<<1, nTree>>>(choiceVector.data(), depth, numLeaves, choices_d);
  cudaDeviceSynchronize();
}

void SilentOTRecver::pprf_expand() {
  std::vector<uint64_t> activeParent(nTree, 0);
  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  GPUvector<OTblock> *inBuffer, *outBuffer;
  GPUvector<OTblock> recvSums(nTree);
  GPUvector<OTblock> *tmp0, *tmp1;
  uint8_t choice;
  uint64_t offsetInVec;

  while(!eventsRecorded);
  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
    outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;
    OTblock *inPtr = inBuffer->data();
    OTblock *outPtr = outBuffer->data();

    uint64_t packedWidth = nTree * width;
    aesLeft.expand_async(outPtr, leftNodes, inPtr, packedWidth, 0, stream[0]);
    aesRight.expand_async(outPtr, rightNodes, inPtr, packedWidth, 1, stream[1]);

    cudaDeviceSynchronize();

    cudaStreamWaitEvent(stream[0], expandEvents.at(d-1));
    cudaStreamWaitEvent(stream[1], expandEvents.at(d-1));

    leftHash.at(d-1).xor_async(choiceHash.at(d-1), stream[0]);
    rightHash.at(d-1).xor_async(choiceHash.at(d-1), stream[1]);

    if (d == depth) {
      leftHash.at(d).xor_async(choiceHash.at(d), stream[0]);
      rightHash.at(d).xor_async(choiceHash.at(d), stream[1]);
    }

    for (uint64_t t = 0; t < nTree; t++) {
      // insert obtained sum into left side or right side
      // and hash to retrieve active node value
      choice = (choices[d-1] >> t) & 1;
      tmp0 = choice == 0 ? &leftHash.at(d-1) : &rightHash.at(d-1);
      tmp1 = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width / 2 + activeParent.at(t);
      cudaMemcpyAsync(tmp1->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[choice]);
      if (d == depth) {
        tmp0 = choice == 0 ? &rightHash.at(d) : &leftHash.at(d);
        tmp1 = choice == 0 ? &rightNodes : &leftNodes;
        cudaMemcpyAsync(tmp1->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[1-choice]);
      }
    }

    leftNodes.sum_async(nTree, width / 2, stream[0]);
    rightNodes.sum_async(nTree, width / 2, stream[1]);

    cudaDeviceSynchronize();
    printf("recver: node val\n");
    print_gpu<<<1, 1>>>((uint8_t*) rightNodes.data(), 16);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();

    // insert active node value obtained from sum into output
    for (uint64_t t = 0; t < nTree; t++) {
      choice = (choices[d-1] >> t) & 1;
      tmp0 = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width + 2 * activeParent.at(t) + choice;
      cudaMemcpyAsync(outPtr + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[choice]);
      if (d == depth) {
        tmp0 = choice == 0 ? &rightNodes : &leftNodes;
        offsetInVec = t * width + 2 * activeParent.at(t) + (1-choice);
        cudaMemcpyAsync(outPtr + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, stream[1-choice]);
      }
      activeParent.at(t) *= 2;
      activeParent.at(t) += 1 - choice;
    }

    cudaDeviceSynchronize();
    printf("recver: vector\n");
    print_gpu<<<1, 1>>>((uint8_t*) outPtr, 64);
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  eventsRecorded = false;
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  puncVector = *outBuffer;
}
