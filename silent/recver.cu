#include "base_ot.h"
#include "roles.h"
#include "event_log.h"
#include <future>

std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

SilentOTRecver::SilentOTRecver(SilentOTConfig config) :
  SilentOT(config), puncVector(2 * numOT), choiceVector(2 * numOT),
  leftBuffer(std::vector<Vec>(depth+1, Vec(mConfig.nTree))),
  rightBuffer(std::vector<Vec>(depth+1, Vec(mConfig.nTree))) {
  silentOTRecvers[mConfig.id] = this;
  while(silentOTSenders[mConfig.id] == nullptr);
  other = silentOTSenders[mConfig.id];
  get_choice_vector();
}

void SilentOTRecver::base_ot() {
  std::vector<std::future<Vec>> workers;
  for (int d = 0; d < depth; d++) {
    workers.push_back(std::async([d, this]() {
      switch (mConfig.baseOT) {
        case SimplestOT_t: return SimplestOT(Recver, d, mConfig.nTree).recv(mConfig.choices[d]);
      }
      return Vec();
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

__global__
void pathfind(blk *choiceVec, uint64_t depth, uint64_t numLeaves, uint64_t *choices) {
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
  pathfind<<<1, mConfig.nTree>>>(choiceVector.data(), depth, numLeaves, choices_d);
  cudaDeviceSynchronize();
}

void SilentOTRecver::pprf_expand() {
  // init hash keys
  uint32_t k0_blk[4] = {3242342};
  uint32_t k1_blk[4] = {8993849};

  Expand *expander;
  switch (mConfig.expander) {
    case AesExpand_t:
      expander = new AesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
  }

  Vec separated(2 * numOT);
  std::vector<uint64_t> activeParent(mConfig.nTree, 0);
  Vec *tmp0;
  uint8_t choice;
  uint64_t offset;

  while(!eventsRecorded);

  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    expander->expand(puncVector, separated, mConfig.nTree * inWidth);
    cudaStreamWaitEvent(0, other->expandEvents.at(d));
    leftBuffer.at(d).xor_d(choiceHash.at(d));
    rightBuffer.at(d).xor_d(choiceHash.at(d));
    if (d == depth-1) {
      leftBuffer.at(d+1).xor_d(choiceHash.at(d));
      rightBuffer.at(d+1).xor_d(choiceHash.at(d));
    }

    uint64_t choices = mConfig.choices[d];
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      // insert obtained sum into left side or right side
      // and sum together to retrieve active node value
      choice = choices & 1;
      tmp0 = choice == 0 ? &leftBuffer.at(d) : &rightBuffer.at(d);
      offset = choice * (mConfig.nTree * inWidth) + t * inWidth + activeParent.at(t);
      cudaMemcpy(separated.data(offset), tmp0->data(t), sizeof(blk), cudaMemcpyDeviceToDevice);
      if (d == depth-1) {
        tmp0 = choice == 0 ? &leftBuffer.at(d+1) : &rightBuffer.at(d+1);
        offset = (1-choice) * (mConfig.nTree * inWidth) + t * inWidth + activeParent.at(t);
        cudaMemcpy(separated.data(offset), tmp0->data(t), sizeof(blk), cudaMemcpyDeviceToDevice);
      }
      choices >>= 1;
    }

    separated.sum(2 * mConfig.nTree, inWidth);
    uint64_t outWidth = 2 * inWidth;
    // insert active node value obtained from sum into output
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      choice = (mConfig.choices[d] >> t) & 1;
      offset = t * outWidth + 2 * activeParent.at(t) + choice;
      cudaMemcpy(puncVector.data(offset), separated.data(choice * mConfig.nTree + t), sizeof(blk), cudaMemcpyDeviceToDevice);

      if (d == depth-1) {
        offset = t * outWidth + 2 * activeParent.at(t) + (1-choice);
        cudaMemcpy(puncVector.data(offset), separated.data((1-choice) * mConfig.nTree + t), sizeof(blk), cudaMemcpyDeviceToDevice);
      }
      activeParent.at(t) *= 2;
      activeParent.at(t) += 1 - choice;
    }
  }

  eventsRecorded = false;
  cudaDeviceSynchronize();

  delete expander;
}

void SilentOTRecver::lpn_compress() {
  switch (mConfig.compressor) {
    case QuasiCyclic_t:
      QuasiCyclic code(2 * numOT, numOT);
      code.encode(puncVector);
      code.encode(choiceVector);
    // case ExpandAccumulate:
  }
}
