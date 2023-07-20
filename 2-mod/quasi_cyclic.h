#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufftXt.h>
#include "gpu_vector.h"

class QuasiCyclic {
private:
  curandGenerator_t prng;
  const uint64_t rows = 128;
  int mRole;
  uint64_t mIn, mOut, nBlocks, n2Blocks, n64;
  cufftHandle aPlan, bPlan, cPlan;
public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out);
  virtual ~QuasiCyclic();
  void encode(GPUvector<OTblock> &vector);
};

#endif
