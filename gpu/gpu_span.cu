#include <stdexcept>
#include "gpu_span.h"

Span::Span(Mat &data, std::vector<uint64_t> start, std::vector<uint64_t> end) : obj(data) {
  if (data.dims().size() != start.size() || data.dims().size() != end.size())
    throw std::invalid_argument("Span::Span() start end dim does not match matrix dim\n");
  if (start >= data.dims() || end > data.dims())
    throw std::invalid_argument("Span::Span() range exceeds matrix dim\n");
  if (start >= end)
    throw std::invalid_argument("Span::Span() end exceeds start\n");

  this->start = start;
  for (int d = 0; d < data.dims().size(); d++) {
    this->range.push_back(end.at(d) - start.at(d));
  }
}

std::vector<uint64_t> _vector_sum(std::vector<uint64_t> a, std::vector<uint64_t> b) {
  std::vector<uint64_t> c;
  for(int d = 0; d < a.size(); d++) {
    c.push_back(a.at(d) + b.at(d));
  }
  return c;
}

blk* Span::data() const {
  return obj.data(start);
}

blk* Span::data(std::vector<uint64_t> i) const {
  if (i >= range)
    throw std::invalid_argument("Span::data() index exceeds Span dim\n");
  
  return obj.data(_vector_sum(start, i));
};

void Span::set(std::vector<uint64_t> i, blk &val) {
  if (i >= range)
    throw std::invalid_argument("Span::set() Index exceeds Span dim\n");

  obj.set(val, _vector_sum(start, i));
}
