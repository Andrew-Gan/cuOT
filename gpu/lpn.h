#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include "gpu_vector.h"

enum LPNType { QuasiCyclic_t, ExpandAccumulate_t };

class Lpn {
public:
  virtual void encode(Vec &vector) = 0;
};

class QuasiCyclic : public Lpn {
private:
  Role mRole;
  curandGenerator_t prng;
  const uint64_t rows = 8*sizeof(OTblock);
  uint64_t mIn, mOut;
  void *workArea = nullptr;
  int fftsizeLog = -1;
  cufftHandle aPlan, bPlan, cPlan;
  cufftReal *a64_poly, *b64_poly, *c64_poly;
  cufftComplex *a64_fft, *b64_fft, *c64_fft;

public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out);
  virtual ~QuasiCyclic();
  void encode(Vec &vector);
};

class ExpandAccumulate : public Lpn {
public:
  ExpandAccumulate(Role role, uint64_t in, uint64_t out);
  virtual ~ExpandAccumulate();
  void encode(Vec &vector);
};

#endif
