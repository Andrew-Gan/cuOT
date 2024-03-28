#include <stdexcept>
#include "gpu_span.h"

Span::Span(Mat &data) : obj(data) {
  start = std::vector<uint64_t>(data.dims().size(), 0);
  range = data.dims();
}

Span::Span(Mat &data, std::vector<uint64_t> start) : obj(data) {
  if (start.size() != data.dims().size()) {
    throw std::invalid_argument("Span::Span start and matrix have differing num dims");
  }
  this->start = start;
  for (uint64_t d = 0; d < data.dims().size(); d++) {
    range.push_back(data.dim(d) - start.at(d));
  }
}

Span::Span(Mat &data, std::vector<uint64_t> start, std::vector<uint64_t> end) : obj(data) {
  if (data.dims().size() != start.size() || data.dims().size() != end.size())
    throw std::invalid_argument("Span::Span start end dim does not match matrix dim");
  if (start >= data.dims() || end > data.dims())
    throw std::invalid_argument("Span::Span range exceeds matrix dim\n");
  if (start >= end)
    throw std::invalid_argument("Span::Span end exceeds start\n");

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

blk* Span::data(std::vector<uint64_t> i) const {
  if (i >= range)
    throw std::invalid_argument("Span::data index exceeds Span dim");
  
  return obj.data(_vector_sum(start, i));
};

void Span::set(std::vector<uint64_t> i, blk &val) {
  if (i >= range)
    throw std::invalid_argument("Span::set Index exceeds Span dim");

  obj.set(val, _vector_sum(start, i));
}
