#include "rand.h"

GPUMatrix<OTBlock> init_rand(curandGenerator_t &prng, uint64_t height, uint64_t width) {
  GPUMatrix<OTBlock> randMatrix;
  randMatrix.rows = height;
  randMatrix.cols = width;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  cudaMalloc(&randMatrix.block.data_d, height * width / 8);
  return randMatrix;
}

void gen_rand(curandGenerator_t prng, GPUMatrix<OTBlock> randMatrix) {
  curandGenerateUniform(prng, (float*) randMatrix.block.data_d, randMatrix.rows * randMatrix.cols / 32);
}

void del_rand(curandGenerator_t prng, GPUMatrix<OTBlock> randMatrix) {
  curandDestroyGenerator(prng);
  cudaFree(randMatrix.block.data_d);
}
