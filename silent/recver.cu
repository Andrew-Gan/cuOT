#include "base_ot.h"
#include "roles.h"
#include "event_log.h"
#include <future>

#include "gpu_ops.h"
#include "gpu_tests.h"

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

__device__
void blk_xor(blk *a, blk *b) {
  for (int i = 0; i < 4; i++) {
    a->data[i] ^= b->data[i];
  }
}

__global__
void fill_punc_tree(blk *leftSum, blk *rightSum, uint64_t outWidth, uint64_t *activeParent,
  uint64_t choice, blk *puncSum, blk *layer) {
  
  uint64_t numTree = gridDim.x * blockDim.x;
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  int c = (choice >> t) & 1;
  blk *fullSum = c == 0 ? leftSum : rightSum;

  blk val = layer[t * outWidth + 2 * activeParent[t] + c];
  blk_xor(&val, &fullSum[t]);
  blk_xor(&val, &puncSum[c * numTree + t]);
  layer[t * outWidth + (2 * activeParent[t] + c)] = val;
  activeParent[t] = 2 * activeParent[t] + (1-c);
}

void SilentOTRecver::pprf_expand() {
  Expand *expander;
  switch (mConfig.expander) {
    case AesExpand_t:
      expander = new AesExpand((uint8_t*)mConfig.leftKey, (uint8_t*)mConfig.rightKey);
  }

  Vec separated(2 * numOT);
  uint64_t *activeParent;
  cudaMalloc(&activeParent, mConfig.nTree * sizeof(uint64_t));
  cudaMemset(activeParent, 0, mConfig.nTree * sizeof(uint64_t));
  while(!eventsRecorded);

  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    expander->expand(puncVector, separated, mConfig.nTree * inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);
    cudaStreamWaitEvent(0, other->expandEvents.at(d));

    leftBuffer.at(d) = other->leftHash.at(d);
    rightBuffer.at(d) = other->rightHash.at(d);

    leftBuffer.at(d).xor_d(choiceHash.at(d));
    rightBuffer.at(d).xor_d(choiceHash.at(d));

    fill_punc_tree<<<1, mConfig.nTree>>>(leftBuffer.at(d).data(),
      rightBuffer.at(d).data(), 2 * inWidth, activeParent,
      mConfig.choices[d], separated.data(), puncVector.data());
    
    if (d == depth-1) {
      leftBuffer.at(d+1) = other->leftHash.at(d+1);
      rightBuffer.at(d+1) = other->rightHash.at(d+1);
      leftBuffer.at(d+1).xor_d(choiceHash.at(d));
      rightBuffer.at(d+1).xor_d(choiceHash.at(d));
      fill_punc_tree<<<1, mConfig.nTree>>>(leftBuffer.at(d+1).data(),
        rightBuffer.at(d+1).data(), 2 * inWidth, activeParent,
        mConfig.choices[d], separated.data(), puncVector.data());
    }
  }

  eventsRecorded = false;
  cudaDeviceSynchronize();

  cudaFree(activeParent);
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
