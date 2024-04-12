#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"
#include "gpu_tests.h"

std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

SilentOTRecver::SilentOTRecver(SilentConfig config) : SilentOT(config) {
  mRole = Recver;
  cudaSetDevice(mConfig.id);
  silentOTRecvers[mConfig.id] = this;
  if(silentOTSenders[mConfig.id] == nullptr) {
    std::runtime_error(
      "SilentOTRecver::SilentOTRecver sender with same id not initialised\n"
    );
  }
  other = silentOTSenders[mConfig.id];

  m0 = std::vector<Mat>(depth+1, Mat({mConfig.nTree}));
  m1 = std::vector<Mat>(depth+1, Mat({mConfig.nTree}));
  
  puncVector = new Mat({numOT, 1});
  buffer = new Mat(puncVector->dims());
  cudaMalloc(&activeParent, mConfig.nTree * sizeof(uint64_t));
  separated.resize({numOT});
  switch (mConfig.pprf) {
    case Aes_t:
      expander = new Aes(mConfig.leftKey, mConfig.rightKey);
  }

  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  b64.resize({rowsPerGPU, 2 * numOT / BLOCK_BITS});
  b64.clear();
  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Recver, 2 * numOT, numOT, BLOCK_BITS / mConfig.gpuPerParty);
  }

  cudaMalloc(&puncPos, mConfig.nTree * sizeof(uint64_t));
  get_choice_vector();
  lpn->encode_sparse(choiceVector, puncPos, mConfig.nTree);
}

SilentOTRecver::~SilentOTRecver() {
  cudaSetDevice(mConfig.id);
  cudaFree(puncPos);
  cudaFree(activeParent);
  delete expander;
  delete puncVector;
  delete buffer;
  delete lpn;
  silentOTRecvers[mConfig.id] = nullptr;
}


void SilentOTRecver::base_ot() {
  cudaSetDevice(mConfig.id);
  std::vector<std::future<Mat>> workers;
  for (int d = 0; d < depth; d++) {
    workers.push_back(std::async([d, this]() {
      cudaSetDevice(mConfig.id);
      switch (mConfig.baseOT) {
        case SimplestOT_t:
          return SimplestOT(Recver, this->mConfig.id*this->depth+d, mConfig.nTree).recv(mConfig.choices[d]);
      }
      return Mat();
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    mc.push_back(res);
  }
}

__global__
void choice_bits_to_pos(uint64_t *choiceVector, uint64_t *choiceBits, uint64_t depth) {
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t id = 0;
  for (uint64_t d = 0; d < depth; d++) {
    id *= 2;
    id += 1-(choiceBits[d] >> t & 1);
  }
  choiceVector[t] = id + t * (1 << depth);
}

void SilentOTRecver::get_choice_vector() {
  uint64_t *choices_d;
  cudaMalloc(&choices_d, depth * sizeof(*choices_d));
  cudaMemcpy(choices_d, mConfig.choices, depth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  choice_bits_to_pos<<<1, mConfig.nTree>>>(puncPos, choices_d, depth);
  cudaDeviceSynchronize();
  cudaFree(choices_d);
}

__global__
void fill_tree(blk *leftSum, blk *rightSum, uint64_t outWidth, uint64_t *activeParent,
  uint64_t choice, blk *puncSum, blk *layer, bool finalLayer) {

  uint64_t numTree = gridDim.x * blockDim.x;
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  int c = (choice >> t) & 1;
  blk *fullSum = c == 0 ? leftSum : rightSum;
  uint64_t fillIndex = t * outWidth;
  fillIndex += finalLayer ? activeParent[t] : 2 * activeParent[t] + c;
  blk val = layer[fillIndex];
  uint64_t puncOffset = (finalLayer ? 1-c : c) * numTree + t;
  for (int i = 0; i < 4; i++)
    val.data[i] ^= fullSum[t].data[i] ^ puncSum[puncOffset].data[i];
  layer[fillIndex] = val;
  if (!finalLayer)
    activeParent[t] = 2 * activeParent[t] + (1-c);
}

void SilentOTRecver::get_punctured_key() {
  cudaSetDevice(mConfig.id);
  for (uint64_t d = 0; d < depth+1; d++) {
    m0.at(d) = other->m0.at(d);
    m1.at(d) = other->m1.at(d);
  }
}

void SilentOTRecver::seed_expand() {
  cudaSetDevice(mConfig.id);
  Log::mem(Recver, SeedExp);
  Mat *input;
  Mat *output;
  cudaMemset(activeParent, 0, mConfig.nTree * sizeof(uint64_t));

  input = buffer;
  output = puncVector;
  for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
    std::swap(input, output);
    expander->expand(*input, *output, separated, mConfig.nTree*inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);

    m0.at(d).xor_d(mc.at(d));
    m1.at(d).xor_d(mc.at(d));

    fill_tree<<<1, mConfig.nTree>>>(m0.at(d).data(), m1.at(d).data(),
      2 * inWidth, activeParent, mConfig.choices[d],
      separated.data(), output->data(), false);
    
    if (d == depth-1) {
      m0.at(d+1).xor_d(mc.at(d));
      m1.at(d+1).xor_d(mc.at(d));

      fill_tree<<<1, mConfig.nTree>>>(m0.at(d+1).data(), m1.at(d+1).data(),
        2 * inWidth, activeParent, mConfig.choices[d],
        separated.data(), output->data(), true);
    }
  }
  Log::mem(Recver, SeedExp);

  puncVector = output;
  buffer = input;
}

void SilentOTRecver::dual_lpn() {
  cudaSetDevice(mConfig.id);
  Log::mem(Recver, LPN);
  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  puncVector->bit_transpose();
  Span b64(*puncVector, {mConfig.id*rowsPerGPU, 0}, {(mConfig.id+1)*rowsPerGPU, 0});
  lpn->encode_dense(b64);
  puncVector->resize({puncVector->dim(0), numOT / BLOCK_BITS});
  puncVector->bit_transpose();
  cudaDeviceSynchronize();
  Log::mem(Recver, LPN);
}
