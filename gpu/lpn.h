#ifndef __LPN_H__
#define __LPN_H__

#include "gpu_matrix.h"
#include "gpu_span.h"

enum LPNType { QuasiCyclic_t, PrimalLpn_t };

class DualLpn {
public:
  virtual ~DualLpn() {}
  virtual void encode_dense(Mat &b64) = 0;
  virtual void encode_sparse(Mat &out, uint64_t *sparsePos, int weight) {}
};

class PrimalLpn {
public:
  virtual ~PrimalLpn() {}
  virtual void encode(Span &nn, Span &kk) = 0;
};

#endif
