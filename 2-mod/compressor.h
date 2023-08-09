#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufftXt.h>
#include "gpu_vector.h"

enum CompressType { QuasiCyclic_t, ExpandAccumulate_t };

class Compressor {
public:
  virtual void encode(GPUvector<OTblock> &vector) = 0;
};

class QuasiCyclic : public Compressor {
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
  void encode(GPUvector<OTblock> &vector);
};

class ExpandAccumulate : public Compressor {
public:
  ExpandAccumulate(Role role, uint64_t in, uint64_t out);
  virtual ~ExpandAccumulate();
  void encode(GPUvector<OTblock> &vector);
};

#endif
