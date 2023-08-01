#include "base_ot.h"
#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

SilentOTRecver::SilentOTRecver(SilentOTConfig config) :
  SilentOT(config), puncVector(2 * numOT), choiceVector(2 * numOT),
  leftBuffer(std::vector<GPUvector<OTblock>>(depth+1, GPUvector<OTblock>(mConfig.nTree))),
  rightBuffer(std::vector<GPUvector<OTblock>>(depth+1, GPUvector<OTblock>(mConfig.nTree))) {
  expandEvents.resize(depth);
  for (auto &event : expandEvents) {
    cudaEventCreate(&event);
  }
  silentOTRecvers[mConfig.id] = this;
  while(silentOTSenders[mConfig.id] == nullptr);
  other = silentOTSenders[mConfig.id];
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
  mult_compress();
  Log::end(Recver, Compress);
}

void SilentOTRecver::base_ot() {
  std::vector<std::future<GPUvector<OTblock>>> workers;
  for (int d = 0; d <= depth; d++) {
    workers.push_back(std::async([d, this]() {
      switch (mConfig.baseOT) {
        case SimplestOT_t: return SimplestOT(Recver, d, mConfig.nTree).recv(mConfig.choices[d]);
      }
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
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
  choiceVector.clear();
  cudaMalloc(&choices_d, depth * sizeof(*choices_d));
  cudaMemcpy(choices_d, mConfig.choices, depth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  pathToChoice<<<1, mConfig.nTree>>>(choiceVector.data(), depth, numLeaves, choices_d);
  cudaDeviceSynchronize();
}

void SilentOTRecver::pprf_expand() {
  // init hash keys
  uint32_t k0_blk[4] = {3242342};
  uint32_t k1_blk[4] = {8993849};

  Expander *expander;
  switch (mConfig.expander) {
    case AesHash_t:
      expander = new AesHash((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  }

  // init buffers
  GPUvector<OTblock> bufferA(2 * numOT), bufferB(2 * numOT);
  GPUvector<OTblock> leftNodes(numOT), rightNodes(numOT);

  std::vector<uint64_t> activeParent(mConfig.nTree, 0);
  cudaStream_t s;
  cudaStreamCreate(&s);
  GPUvector<OTblock> *inBuffer, *outBuffer;
  GPUvector<OTblock> recvSums(mConfig.nTree);
  GPUvector<OTblock> *tmp0, *tmp1;
  uint8_t choice;
  uint64_t offsetInVec;

  Log::end(Recver, Expand);
  while(!eventsRecorded);
  Log::start(Recver, Expand);

  for (uint64_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    inBuffer = (d % 2 == 1) ? &bufferA : &bufferB;
    outBuffer = (d % 2 == 1) ? &bufferB : &bufferA;

    uint64_t packedWidth = mConfig.nTree * width;
    expander->expand_async(*outBuffer, leftNodes, rightNodes, *inBuffer, packedWidth, s);

    cudaStreamWaitEvent(s, expandEvents.at(d-1));

    leftBuffer.at(d-1).xor_async(choiceHash.at(d-1), s);
    rightBuffer.at(d-1).xor_async(choiceHash.at(d-1), s);

    if (d == depth) {
      leftBuffer.at(d).xor_async(choiceHash.at(d), s);
      rightBuffer.at(d).xor_async(choiceHash.at(d), s);
    }

    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      // insert obtained sum into left side or right side
      // and sum together to retrieve active node value
      choice = (mConfig.choices[d-1] >> t) & 1;
      tmp0 = choice == 0 ? &leftBuffer.at(d-1) : &rightBuffer.at(d-1);
      tmp1 = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width / 2 + activeParent.at(t);
      cudaMemcpyAsync(tmp1->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, s);
      if (d == depth) {
        tmp0 = choice == 0 ? &rightBuffer.at(d) : &leftBuffer.at(d);
        tmp1 = choice == 0 ? &rightNodes : &leftNodes;
        cudaMemcpyAsync(tmp1->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, s);
      }
    }

    leftNodes.sum_async(mConfig.nTree, width / 2, s);
    rightNodes.sum_async(mConfig.nTree, width / 2, s);

    // insert active node value obtained from sum into output
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      choice = (mConfig.choices[d-1] >> t) & 1;
      tmp0 = choice == 0 ? &leftNodes : &rightNodes;
      offsetInVec = t * width + 2 * activeParent.at(t) + choice;
      cudaMemcpyAsync(outBuffer->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, s);

      if (d == depth) {
        tmp0 = choice == 0 ? &rightNodes : &leftNodes;
        offsetInVec = t * width + 2 * activeParent.at(t) + (1-choice);
        cudaMemcpyAsync(outBuffer->data() + offsetInVec, tmp0->data() + t, sizeof(OTblock), cudaMemcpyDeviceToDevice, s);
      }
      activeParent.at(t) *= 2;
      activeParent.at(t) += 1 - choice;
    }
  }
  
  eventsRecorded = false;
  cudaDeviceSynchronize();
  cudaStreamDestroy(s);
  puncVector = *outBuffer;

  delete expander;
}

void SilentOTRecver::mult_compress() {
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      QuasiCyclic code(Recver, 2 * numOT, numOT);
      code.encode(puncVector);
      code.encode(choiceVector);
    // case ExpandAccumulate:
  }
}
