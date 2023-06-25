#include "rand.h"

Matrix init_rand(curandGenerator_t &prng, uint64_t height, uint64_t width) {
  Matrix randMatrix;
  randMatrix.rows = height;
  randMatrix.cols = width;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  cudaMalloc(&randMatrix.data, height * width / 8);
  return randMatrix;
}

void gen_rand(curandGenerator_t prng, Matrix randMatrix) {
  curandGenerateUniform(prng, (float*) randMatrix.data, randMatrix.rows * randMatrix.cols / 32);
}

void del_rand(curandGenerator_t prng, Matrix randMatrix) {
  curandDestroyGenerator(prng);
  cudaFree(randMatrix.data);
}
