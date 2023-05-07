#include "rand.h"
#include <curand_kernel.h>

curandGenerator_t prng;
Matrix d_randMatrix;

Matrix gen_rand(size_t height, size_t width) {
  static bool isInit = false;
  d_randMatrix.rows = height;
  d_randMatrix.cols = width;

  if (!isInit) {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, clock());
    cudaMalloc(&d_randMatrix.data, height * width / 8);
    isInit = true;
  }

  curandGenerateUniform(prng, (float*) d_randMatrix.data, width * height / 32);
  return d_randMatrix;
}

void del_rand() {
  curandDestroyGenerator(prng);
  cudaFree(d_randMatrix.data);
}
