#include "silent_ot.h"
#include <future>

#include "logger.h"
#include "gpu_tests.h"
#include "gpu_define.h"
#include "gpu_ops.h"
#include <cryptoTools/Crypto/RandomOracle.h>

blk* SOTRecver::mc_h = nullptr;
std::array<std::atomic<SOTRecver*>, 16> SOTRecvers;

SOTRecver::SOTRecver(SilentConfig config) : SOT(config) {
  mRole = Recver;
  mGPU = mConfig.gpuPerParty + mConfig.id;
  cudaSetDevice(mGPU);
  SOTRecvers[mConfig.id] = this;
  if(SOTSenders[mConfig.id] == nullptr)
    throw std::runtime_error("SOTRecver::SOTRecver sender not initialised\n");
  other = SOTSenders[mConfig.id];

  m0.resize({mDepth+1, mConfig.nTree});
  m1.resize({mDepth+1, mConfig.nTree});
  mc.resize({mDepth, mConfig.nTree});
  cudaMalloc(&activeParent, mConfig.nTree * sizeof(uint64_t));

#ifdef USE_COALESCED_TREE_EXPANSION
  puncVector = new Mat({numOT, 1});
  buffer = new Mat(puncVector->dims());
  sep = new Mat({numOT});
#else
  puncVector = new Mat[mConfig.nTree];
  buffer = new Mat[mConfig.nTree];
  sep = new Mat[mConfig.nTree];
  for (int t = 0; t < mConfig.nTree; t++) {
    puncVector[t].resize({numOT / mConfig.nTree, 1});
    buffer[t].resize({numOT / mConfig.nTree, 1});
    sep[t].resize({numOT / mConfig.nTree});
  }
#endif

  switch (mConfig.pprf) {
    case Aes_t:
      expander = new Aes(mConfig.leftKey, mConfig.rightKey);
  }
  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Recver, 2 * numOT, numOT, BLOCK_BITS / mConfig.gpuPerParty);
  }

  get_choice_vector();

  if (mConfig.id == 0)
    SOTRecver::mc_h = new blk[mDepth * mConfig.nTree];

  CHECK_CUDA("SOTRecver::SOTRecver")
}

SOTRecver::~SOTRecver() {
  cudaSetDevice(mGPU);

#ifdef USE_COALESCED_TREE_EXPANSION
  delete puncVector;
  delete buffer;
#else
  delete[] puncVector;
  delete[] buffer;
#endif

  delete expander;
  delete lpn;
  if (puncPos) cudaFree(puncPos);
  cudaFree(activeParent);
  if (mConfig.id == 0) {
    delete[] SOTRecver::mc_h;
  }
  SOTRecvers[mConfig.id] = nullptr;
}


void SOTRecver::base_ot() {
  cudaSetDevice(mGPU);
  std::vector<std::future<void>> workers;
  for (uint64_t d = 0; d < mDepth; d++) {
    workers.push_back(std::async([d, this](){
      SimplestOT bOT(Recver, d, mConfig.nTree);
      bOT.recv(SOTRecver::mc_h+d*mConfig.nTree, mConfig.choices[d]);
    }));
  }
  for (auto &t : workers) {
    t.get();
  }
}

__global__
void choice_bits_to_pos(uint64_t *choiceVector, uint64_t *choiceBits, uint64_t mDepth) {
  uint64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t id = 0;
  for (uint64_t d = 0; d < mDepth; d++) {
    id *= 2;
    id += 1-(choiceBits[d] >> t & 1);
  }
  choiceVector[t] = id + t * (1 << mDepth);
}

void SOTRecver::get_choice_vector() {
  if (puncPos == nullptr)
    cudaMalloc(&puncPos, mConfig.nTree * sizeof(*puncPos));
  uint64_t *choices_d;
  cudaMalloc(&choices_d, mDepth * sizeof(*choices_d));
  cudaMemcpy(choices_d, mConfig.choices, mDepth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  choice_bits_to_pos<<<1, mConfig.nTree>>>(puncPos, choices_d, mDepth);
  cudaError_t err = cudaDeviceSynchronize();
  CHECK_CUDA("SOTRecver::get_choice_vector")
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
    val.data_32[i] ^= fullSum[t].data_32[i] ^ puncSum[puncOffset].data_32[i];
  layer[fillIndex] = val;
  if (!finalLayer)
    activeParent[t] = 2 * activeParent[t] + (1-c);
}

void SOTRecver::get_punc_key() {
  cudaSetDevice(mGPU);
  m0 = other->m0;
  m1 = other->m1;
}

void SOTRecver::seed_exp() {
  cudaSetDevice(mGPU);

  cudaMemcpy(mc.data(), SOTRecver::mc_h, mc.size_bytes(), cudaMemcpyHostToDevice);  
  cudaMemset(activeParent, 0, mConfig.nTree * sizeof(uint64_t));

  Mat *input = buffer;
  Mat *output = puncVector;
  uint64_t numBytes = mConfig.nTree * sizeof(blk);

  for (uint64_t d = 0, inWidth = 1; d < mDepth; d++, inWidth *= 2) {
    std::swap(input, output);
#ifdef USE_COALESCED_TREE_EXPANSION
    expander->expand(*input, *output, *sep, mConfig.nTree*inWidth);
    sep->sum(2 * mConfig.nTree, inWidth);

    gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    fill_tree<<<1, mConfig.nTree>>>(m0.data({d, 0}), m1.data({d, 0}), 2 * inWidth,
      activeParent, mConfig.choices[d], sep->data(), output->data(), false);
    if (d == mDepth-1) {
      gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
      gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
      fill_tree<<<1, mConfig.nTree>>>(m0.data({d+1, 0}), m1.data({d+1, 0}),
        2 * inWidth, activeParent, mConfig.choices[d],
        sep->data(), output->data(), true);
    }
#else
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      expander->expand(input[t], output[t], sep[t], inWidth);
      uint64_t block = std::min(1024UL, 2 * inWidth);
      uint64_t grid = (2 * inWidth + block - 1) / block;
      separator<<<grid, block>>>(sep[t].data(), output[t].data());
      sep[t].sum(2, inWidth);
    }

    gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    if (d == mDepth-1) {
      gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
      gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    }
    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      fill_tree<<<1, 1>>>(m0.data({d, t}), m1.data({d, t}), 2*inWidth, activeParent+t,
        mConfig.choices[d] >> t, sep[t].data(), output[t].data(), false);
      if (d == mDepth-1) {
        fill_tree<<<1, 1>>>(m0.data({d+1, t}), m1.data({d+1, t}), 2*inWidth,
          activeParent+t, mConfig.choices[d] >> t, sep[t].data(), output[t].data(), true);
      }
    }
#endif // USE_COALESCED_TREE_EXPANSION
  }

  puncVector = output;
  buffer = input;
  cudaError_t err = cudaDeviceSynchronize();
  CHECK_CUDA("SOTRecver::seed_exp")
}

void SOTRecver::dual_lpn() {
  cudaSetDevice(mGPU);
  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  puncVector->bit_transpose(mConfig.id*rowsPerGPU, (mConfig.id+1)*rowsPerGPU);
  lpn->encode_dense(*puncVector);
  puncVector->bit_transpose();
  lpn->encode_sparse(choiceVector, puncPos, mConfig.nTree);
  cudaError_t err = cudaDeviceSynchronize();
  CHECK_CUDA("SOTRecver::dual_lpn")
}
