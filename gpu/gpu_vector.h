#ifndef __GPU_VECTOR_H__
#define __GPU_VECTOR_H__

#include <iostream>
#include "gpu_matrix.h"

class Span;

class Vec : public Mat {
public:
  using GPUdata::xor_d;
  using Mat::data;

  Vec() {}
  Vec(uint64_t len) : Mat({len}) {}
  uint64_t size() const { return dim(0); }
  blk* data(uint64_t i) const;
  void set(uint64_t i, blk &val) { Mat::set(val, {i}); }
  void resize(uint64_t len) { Mat::resize({len}); }
  void sum(uint64_t nPartition, uint64_t blkPerPart);
  void xor_d(Vec &rhs, uint64_t offs = 0);
  Span span(uint64_t start = 0, uint64_t end = 0);
};

class Span {
private:
  uint64_t start, range;
  Vec &obj;

public:
  Span(Vec &data, uint64_t start, uint64_t end);
  uint64_t size() const { return range; }
  blk* data(uint64_t i = 0) const;
  void set(uint64_t i, blk &val);
  void operator=(const Span& other);
};

#endif
