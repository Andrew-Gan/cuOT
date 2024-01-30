#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufft.h>
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
  Mat b64;
  void *workArea;
  int mPartition = 1;
  int fftsizeLog = -1;
  cufftHandle bPlan, cPlan;
  cufftReal *b64_poly;
  cufftComplex *a64_fft, *b64_fft;
  Mat cModP1;

public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out, int partition);
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
