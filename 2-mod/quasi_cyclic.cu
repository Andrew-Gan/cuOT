#include "quasi_cyclic.h"
#include <cmath>

#define DENSITY 4096 // out of matrix numCols

QuasiCyclic::QuasiCyclic(uint64_t in, uint64_t out) : mIn(in), mOut(out) {
  if (in == 0 || out == 0) return;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  cudaMalloc(&nonZeroPos, DENSITY * sizeof(float));
  curandGenerateUniform(prng, (float*) nonZeroPos, DENSITY);
}

QuasiCyclic::~QuasiCyclic() {
  if (numCols == 0) return;
  curandDestroyGenerator(prng);
  if (nonZeroPos) cudaFree(nonZeroPos);
}

__global__
void dot_product(float *nonZeroPos, uint64_t cols, OTBlock *vec) {
  OTBlock res;
  uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t rand = 0;

  for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
    res.data[j] = 0;
  }

  for (uint64_t i = 0; i < DENSITY; i++) {
    rand = (uint64_t) (nonZeroPos[i] * (cols-1));
    rand = (rand + row) % cols;
    for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
      res.data[j] ^= vec[rand].data[j];
    }
  }
  __syncthreads();

  for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
    vec[row].data[j] = res.data[j];
  }
}

void QuasiCyclic::encode(GPUBlock &vector) {
  uint64_t firstMatrixNumRows = (1 << 10);
  uint64_t nB = firstMatrixNumRows / 1024;
  dot_product<<<firstMatrixNumRows, 1024>>>(nonZeroPos, numCols, (OTBlock*)vector.data_d);
  cudaDeviceSynchronize();
  uint64_t firstMatrixNumRows = (1 << 10);

  dot_product<<<out, 1024>>>(nonZeroPos, firstMatrixNumRows, (OTBlock*)vector.data_d);
  cudaDeviceSynchronize();
  vector.resize(out * sizeof(OTBlock));
}
