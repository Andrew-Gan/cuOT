#include "quasi_cyclic.h"

QuasiCyclic::QuasiCyclic(uint64_t n) : numCols(n), numRows(n/2) {
  if (n == 0) return;
  printf("n = %lu, numRows = %lu\n", n, numRows);
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  cudaMalloc(&nonZeroPos, (n / 4) * sizeof(float));
  curandGenerateUniform(prng, (float*) nonZeroPos, n / 4);
  nonZeroCount = n / 4;
}

QuasiCyclic::~QuasiCyclic() {
  if (numCols == 0) return;
  curandDestroyGenerator(prng);
  if (nonZeroPos) cudaFree(nonZeroPos);
}

__global__
void dot_product(float *nonZeroPos, uint64_t nonZeroCount, uint64_t n, OTBlock *vec) {
  extern __shared__ OTBlock res[];
  uint64_t bid = blockIdx.x;
  uint64_t tid = threadIdx.x;
  for (int i = 0; i < sizeof(OTBlock) / 4; i++) {
    res[tid].data[i] = 0;
  }

  uint64_t workload = nonZeroCount / blockDim.x;
  for (uint64_t i = tid*workload; i < (tid+1)*workload; i++) {
    uint64_t col = (((uint32_t)(nonZeroPos[i]) % n) - bid) % n;
    // for (int j = 0; j < sizeof(OTBlock) / 4; j++) {
    //   res[tid].data[j] ^= vec[col].data[j];
    // }
  }
  __syncthreads();

  for (int threadGroup = 512; threadGroup >= 1; threadGroup /= 2) {
    if (tid < threadGroup) {
      for (int j = 0; j < sizeof(OTBlock) / 4; j++) {
        res[tid].data[j] ^= vec[tid+threadGroup].data[j];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int j = 0; j < sizeof(OTBlock) / 4; j++) {
      vec[tid].data[j] = res[0].data[j];
    }
  }
}

void QuasiCyclic::encode(GPUBlock &vector) {
  printf("%lu x %lu\n", numRows, numCols);
  return;
  dot_product<<<numRows, 1024, 1024 * sizeof(OTBlock)>>>(nonZeroPos, nonZeroCount, numCols, (OTBlock*)vector.data_d);
  cudaDeviceSynchronize();
  vector.resize(vector.nBytes / 2);
}
