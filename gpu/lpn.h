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
  curandGenerator_t prng[2];
  const uint64_t rows = 8*sizeof(OTblock);
  uint64_t mIn, mOut;
  void *workArea[2];
  int fftsizeLog = -1;
  cufftHandle aPlan[2], bPlan[2], cPlan[2];
  cufftReal *a64_poly[2], *b64_poly[2], *c64_poly[2];
  cufftComplex *a64_fft[2], *b64_fft[2], *c64_fft[2];

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
