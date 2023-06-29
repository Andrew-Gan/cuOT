#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include "gpu_block.h"

class QuasiCyclic {
private:
  curandGenerator_t prng;
  float *nonZeroPos = nullptr;
  uint64_t numRows, numCols;
public:
  QuasiCyclic(uint64_t in, uint64_t out);
  virtual ~QuasiCyclic();
  void encode(GPUBlock &vector);
};

#endif
