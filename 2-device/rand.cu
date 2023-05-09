#include "rand.h"
#include <curand_kernel.h>

curandGenerator_t prng;
Matrix randMatrix_d;

Matrix gen_rand(size_t height, size_t width) {
  static bool isInit = false;
  randMatrix_d.rows = height;
  randMatrix_d.cols = width;

  if (!isInit) {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, clock());
    cudaMalloc(&randMatrix_d.data, height * width / 8);
    isInit = true;
  }

  curandGenerateUniform(prng, (float*) randMatrix_d.data, width * height / 32);
  return randMatrix_d;
}

void del_rand() {
  curandDestroyGenerator(prng);
  cudaFree(randMatrix_d.data);
}
