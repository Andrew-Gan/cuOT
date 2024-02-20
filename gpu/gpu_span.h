#include "gpu_matrix.h"

class Span {
private:
  uint64_t start, range;
  Mat &obj;

public:
  Span(Mat &data, uint64_t start, uint64_t end);
  uint64_t size() const { return range; }
  blk* data(uint64_t i = 0) const;
  void set(uint64_t i, blk &val);
  void operator=(const Span& other);
};