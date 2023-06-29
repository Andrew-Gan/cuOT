#include "quasi_cyclic.h"
#include <cmath>

#define DENSITY 1024 // out of matrix numCols

QuasiCyclic::QuasiCyclic(uint64_t in, uint64_t out) : numCols(in), numRows(out) {
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
void dot_product(float *nonZeroPos, uint64_t threadsPerRow, uint64_t cols, OTBlock *vec) {
  extern __shared__ OTBlock s[];
  uint64_t tid_global = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t tid_local = threadIdx.x;
  uint64_t row = tid_global / threadsPerRow;
  uint64_t innerRow = tid_global % threadsPerRow;
  uint64_t rand = 0;

  for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
    s[tid_local].data[j] = 0;
  }

  uint64_t workload = DENSITY / threadsPerRow;
  for (uint64_t i = innerRow * workload; i < (innerRow+1) * workload; i++) {
    rand = (uint64_t) (nonZeroPos[i] * (cols-1));
    rand = (rand + row) % cols;
    for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
      s[tid_local].data[j] ^= vec[rand].data[j];
    }
  }
  __syncthreads();

  if (innerRow == 0) {
    for (uint64_t i = 0; i < threadsPerRow; i++) {
      for (uint64_t j = 0; j < sizeof(OTBlock) / 4; j++) {
        vec[row].data[j] = s[tid_local].data[j];
      }
    }
  }
}

void QuasiCyclic::encode(GPUBlock &vector) {
  cudaStream_t s;
  cudaStreamCreate(&s);
  uint64_t threadsPerRow = 1;
  uint64_t nB = threadsPerRow * numRows / 1024;
  uint64_t mem = 1024 * sizeof(OTBlock);
  dot_product<<<nB, 1024, mem, s>>>(nonZeroPos, threadsPerRow, numCols, (OTBlock*)vector.data_d);
  cudaDeviceSynchronize();
  cudaStreamDestroy(s);
  vector.resize(vector.nBytes / 2);
}
