#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <curand_kernel.h>
#include <cufft.h>
#include "gpu_matrix.h"

enum LPNType { QuasiCyclic_t, ExpandAccumulate_t };

class Lpn {
public:
  virtual void encode_dense(Mat &b64) = 0;
  virtual void encode_sparse(Mat &out, uint64_t *sparsePos, int weight) = 0;
};

class QuasiCyclic : public Lpn {
private:
  Role mRole;
  curandGenerator_t prng;
  uint64_t mIn, mOut;
  void *workArea;
  int fftsizeLog = -1;
  Mat a64;
  cufftHandle bPlan, cPlan;
  cufftReal *b64_poly, *c64_poly;
  cufftComplex *a64_fft, *b64_fft;
  uint64_t mRows = 8*sizeof(OTblock);
  uint64_t blockFFT[3];
  dim3 gridFFT[3];

public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out, int rows);
  virtual ~QuasiCyclic();
  void encode_dense(Mat &b64);
  void encode_sparse(Mat &out, uint64_t *sparsePos, int weight);
};

class ExpandAccumulate : public Lpn {
public:
  ExpandAccumulate(Role role, uint64_t in, uint64_t out);
  virtual ~ExpandAccumulate();
  void encode_dense(Mat &b64);
  void encode_sparse(Mat &out, uint64_t *sparsePos, int weight);
};

#endif
