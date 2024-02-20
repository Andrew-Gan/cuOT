#include <stdexcept>
#include "gpu_span.h"

Span::Span(Mat &data, uint64_t start, uint64_t end) : obj(data) {
  if (data.dims().size() != 1) 
    throw std::invalid_argument("Span::Span() Only 1D matrix supported\n");
  if (start >= data.size() || end > data.size())
    throw std::invalid_argument("Span::Span() Range exceeds Vec dim\n");

  this->start = start;
  this->range = end - start;
}

blk* Span::data(uint64_t i) const {
  if (i >= range)
    throw std::invalid_argument("Span::data() Index exceeds Span dim\n");

  return obj.data({start + i});
};

void Span::set(uint64_t i, blk &val) {
  if (i >= range)
    throw std::invalid_argument("Span::set() Index exceeds Span dim\n");

  obj.set(val, {start + i});
}

void Span::operator=(const Span& other) {
  if (size() != other.size())
    throw std::invalid_argument("Span::operator=() Unequal span len is unsupported\n");

  cudaMemcpyAsync(data(), other.data(), size()*sizeof(blk), cudaMemcpyDeviceToDevice);
}
