#include "rand.h"

Matrix init_rand(curandGenerator_t &prng, size_t height, size_t width) {
  EventLog::start(MatrixInit);
  Matrix randMatrix;
  randMatrix.rows = height;
  randMatrix.cols = width;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  cudaMalloc(&randMatrix.data, height * width / 8);
  EventLog::end(MatrixInit);
  return randMatrix;
}

void gen_rand(curandGenerator_t prng, Matrix randMatrix) {
  EventLog::start(MatrixRand);
  curandGenerateUniform(prng, (float*) randMatrix.data, randMatrix.rows * randMatrix.cols / 32);
  EventLog::end(MatrixRand);
}

void del_rand(curandGenerator_t prng, Matrix randMatrix) {
  curandDestroyGenerator(prng);
  cudaFree(randMatrix.data);
}
