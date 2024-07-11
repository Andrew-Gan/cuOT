#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"
#include <cryptoTools/Crypto/RandomOracle.h>

blk* SOTRecver::mc_h = nullptr;
std::array<std::atomic<SOTRecver*>, 16> SOTRecvers;

SOTRecver::SOTRecver(SilentConfig config) : SOT(config) {
  mRole = Recver;
  cudaSetDevice(mConfig.id);
  SOTRecvers[mConfig.id] = this;
  if(SOTSenders[mConfig.id] == nullptr) {
    std::runtime_error(
      "SOTRecver::SOTRecver sender with same id not initialised\n"
    );
  }
  other = SOTSenders[mConfig.id];

  m0.resize({mDepth+1,mConfig.nTree});
  m1.resize({mDepth+1,mConfig.nTree});
  mc.resize({mDepth,mConfig.nTree});
  
  puncVector = new Mat({numOT, 1});
  buffer = new Mat(puncVector->dims());
  cudaMalloc(&activeParent, mConfig.nTree * sizeof(uint64_t));
  separated.resize({numOT});
  switch (mConfig.pprf) {
    case Aes_t:
      expander = new Aes(mConfig.leftKey, mConfig.rightKey);
  }

  switch (mConfig.dualLPN) {
    case QuasiCyclic_t:
      lpn = new QuasiCyclic(Recver, 2 * numOT, numOT, BLOCK_BITS / mConfig.gpuPerParty);
  }

  cudaMalloc(&puncPos, mConfig.nTree * sizeof(uint64_t));
  get_choice_vector();
  lpn->encode_sparse(choiceVector, puncPos, mConfig.nTree);

  if (mConfig.id == 0) {
    SOTRecver::mc_h = new blk[mDepth * mConfig.nTree];
  }
}

SOTRecver::~SOTRecver() {
  cudaSetDevice(mConfig.id);
  cudaFree(puncPos);
  cudaFree(activeParent);
  delete expander;
  delete puncVector;
  delete buffer;
  delete lpn;
  if (mConfig.id == 0) {
    delete[] SOTRecver::mc_h;
  }
  SOTRecvers[mConfig.id] = nullptr;
}


void SOTRecver::base_ot() {
  cudaSetDevice(mConfig.id);
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
  uint64_t *choices_d;
  cudaMalloc(&choices_d, mDepth * sizeof(*choices_d));
  cudaMemcpy(choices_d, mConfig.choices, mDepth * sizeof(*choices_d), cudaMemcpyHostToDevice);
  choice_bits_to_pos<<<1, mConfig.nTree>>>(puncPos, choices_d, mDepth);
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

void SOTRecver::get_punc_key() {
  cudaSetDevice(mConfig.id);
  // senders m0, m1 were XORed with base OT values
  m0 = other->m0;
  m1 = other->m1;
}

void SOTRecver::seed_expand() {
  cudaSetDevice(mConfig.id);
  Log::mem(Recver, SeedExp);

  cudaMemcpy(mc.data(), SOTRecver::mc_h, mc.size_bytes(), cudaMemcpyHostToDevice);
  
  Mat *input;
  Mat *output;
  cudaMemset(activeParent, 0, mConfig.nTree * sizeof(uint64_t));

  input = buffer;
  output = puncVector;
  uint64_t numBytes = mConfig.nTree * sizeof(blk);

  for (uint64_t d = 0, inWidth = 1; d < mDepth; d++, inWidth *= 2) {
    std::swap(input, output);
    expander->expand(*input, *output, separated, mConfig.nTree*inWidth);
    separated.sum(2 * mConfig.nTree, inWidth);

    gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
    gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d, 0}), (uint8_t*)mc.data({d, 0}), numBytes);

    fill_tree<<<1, mConfig.nTree>>>(m0.data({d, 0}), m1.data({d, 0}),
      2 * inWidth, activeParent, mConfig.choices[d],
      separated.data(), output->data(), false);
    
    if (d == mDepth-1) {
      gpu_xor<<<1, numBytes>>>((uint8_t*)m0.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);
      gpu_xor<<<1, numBytes>>>((uint8_t*)m1.data({d+1, 0}), (uint8_t*)mc.data({d, 0}), numBytes);

      fill_tree<<<1, mConfig.nTree>>>(m0.data({d+1, 0}), m1.data({d+1, 0}),
        2 * inWidth, activeParent, mConfig.choices[d],
        separated.data(), output->data(), true);
    }
  }
  Log::mem(Recver, SeedExp);

  puncVector = output;
  buffer = input;
}

blk gf128Mul(blk x, blk y) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t mod = 0b10000111;
  uint64_t *shifted = (uint64_t*)&(x[i]);
  uint64_t * ya = (uint64_t*)&y;
  std::array<uint64_t, 2> result0, result1;

  result0[0] = 0;
  result0[1] = 0;
  result1[0] = 0;
  result1[1] = 0;

  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 64; ++j) {
      if (ya[i] & (1ull << j)) {
        result0[0] ^= shifted[0];
        result0[1] ^= shifted[1];
      }

      if (shifted[1] & (1ull << 63)) {
        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
        shifted[0] = (shifted[0] << 1) ^ mod;
      }
      else {
        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
        shifted[0] = shifted[0] << 1;
      }
    }
  }

  return result0;
}

__global__
void gf128Mul(blk *x, blk y, blk *xy1, blk *xy2) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t mod = 0b10000111;
  uint64_t *shifted = (uint64_t*)&(x[i]);
  uint64_t * ya = (uint64_t*)&y;
  std::array<uint64_t, 2> result0, result1;

  result0[0] = 0;
  result0[1] = 0;
  result1[0] = 0;
  result1[1] = 0;

  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 64; ++j) {
      if (ya[i] & (1ull << j)) {
        result0[0] ^= shifted[0];
        result0[1] ^= shifted[1];
      }

      if (shifted[1] & (1ull << 63)) {
        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
        shifted[0] = (shifted[0] << 1) ^ mod;
      }
      else {
        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
        shifted[0] = shifted[0] << 1;
      }
    }
  }

  xy1 ^= result0;
  xy2 ^= result1;
}

void SOTRecver::mal_check() {
  Mat xx({puncVector.size(), 1});
  Mat sum0({1, 1});
  Mat sum1({1, 1});
  Mat mySum({1, 1});
  Mat b({1, 1});
  NoisyVoleSender sender;
  GPUdata theirHash(32);
  GPUdata myHash(32);
  RandomOracle ro(32);

  chl.send(std::move(mMalCheckSeed));
  xx = mMalCheckSeed;
  sum0.clear();
  sum1.clear();

  for (size_t i = 0; i < puncVector.size(); i++) {
    blk low, high;
    xx.gf128Mul(puncVector.at({0, i}), low, high);
    sum0 = sum0 ^ low;
    sum1 = sum1 ^ high;
    xx = xx.gf128Mul(mMalCheckSeed);
  }
  mySum = sum0.gf128Reduce(sum1);

  co_await(sender.send(mMalCheckX, b, prng, mMalCheckOts, chl, {}));
  ro.Update(mySum ^ b[0]);
  ro.Final(myHash);

  co_await(chl.recv(theirHash));

  if (theirHash != myHash)
    throw RTE_LOC;
}

void SOTRecver::dual_lpn() {
  cudaSetDevice(mConfig.id);
  Log::mem(Recver, LPN);
  uint64_t rowsPerGPU = (BLOCK_BITS + mConfig.gpuPerParty - 1) / mConfig.gpuPerParty;
  puncVector->bit_transpose(mConfig.id*rowsPerGPU, (mConfig.id+1)*rowsPerGPU);
  lpn->encode_dense(*puncVector);
  puncVector->bit_transpose();
  cudaDeviceSynchronize();
  Log::mem(Recver, LPN);
}
