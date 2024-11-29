#ifndef __QUASI_CYCLIC_H__
#define __QUASI_CYCLIC_H__

#include <cufft.h>
#include "gpu_define.h"
#include "lpn.h"

class QuasiCyclic : public DualLpn {
private:
  Role mRole;
  uint64_t mIn, mOut;
  void *workArea;
  Mat a;
  cufftHandle bPlan, cPlan;
  float *b_poly, *c_poly;
  float2 *a_fft, *b_fft;
  uint64_t mRows = 8 * sizeof(blk);
  float *a_poly;

public:
  QuasiCyclic(Role role, uint64_t in, uint64_t out, int rows);
  virtual ~QuasiCyclic();
  void encode_dense(Mat &b64);
  void encode_sparse(Mat &out, uint64_t *sparsePos, int weight);
};

#endif
