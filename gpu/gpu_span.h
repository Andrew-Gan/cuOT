#ifndef __GPU_SPAN_H__
#define __GPU_SPAN_H__

#include "gpu_matrix.h"

class Span {
public:
  std::vector<uint64_t> start, range;
  Mat &obj;

public:
  Span(Mat &data, std::vector<uint64_t> start, std::vector<uint64_t> end);
  uint64_t size() const { return obj.size(); }
  blk* data() const;
  blk* data(std::vector<uint64_t> i) const;
  void set(std::vector<uint64_t> i, blk &val);
};

#endif
