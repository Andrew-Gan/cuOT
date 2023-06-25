#ifndef __RAND_GPU_H__
#define __RAND_GPU_H__

#include "util.h"
#include <curand_kernel.h>

Matrix init_rand(curandGenerator_t &prng, uint64_t height, uint64_t width);
void gen_rand(curandGenerator_t prng, Matrix randMatrix);
void del_rand(curandGenerator_t prng, Matrix randMatrix);

#endif
