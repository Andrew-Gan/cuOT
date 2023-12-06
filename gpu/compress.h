#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufftXt.h>
#include "gpu_vector.h"

enum CompressType { QuasiCyclic_t, ExpandAccumulate_t };

class Compress {
public:
  virtual void encode(Vec &vector) = 0;
};

class QuasiCyclic : public Compress {
private:
  curandGenerator_t prng;
  const uint64_t rows = 128;
  int mRole;
  uint64_t mIn, mOut, nBlocks, n2Blocks, n64;
  cufftHandle aPlan, bPlan, cPlan;
  cufftComplex *a64_fft;
public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out);
  virtual ~QuasiCyclic();
  void encode(Vec &vector);
};

class ExpandAccumulate : public Compress {
public:
  ExpandAccumulate(Role role, uint64_t in, uint64_t out);
  virtual ~ExpandAccumulate();
  void encode(Vec &vector);
};

#endif
