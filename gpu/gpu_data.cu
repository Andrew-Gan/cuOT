#include <iomanip>
#include <fstream>
#include <mutex>

#include "gpu_data.h"
#include "gpu_ops.h"
#include "gpu_tests.h"

GPUdata::GPUdata(uint64_t n) : mNBytes(n) {
  cudaError_t err = cudaMalloc(&mPtr, n);
  check_call("GPUdata:GPUdata\n");
}

GPUdata::GPUdata(const GPUdata &blk) : GPUdata(blk.size_bytes()) {
  cudaMemcpy(mPtr, blk.data(), mNBytes, cudaMemcpyDeviceToDevice);
}

GPUdata::~GPUdata() {
  if (mPtr != nullptr) cudaFree(mPtr);
}

GPUdata& GPUdata::operator&=(const GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  gpu_and<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  check_call("GPUdata::operator&=\n");
  return *this;
}

GPUdata& GPUdata::operator^=(const GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  check_call("GPUdata::operator^=\n");
  return *this;
}

GPUdata& GPUdata::operator=(const GPUdata &rhs) {
  if (mNBytes != rhs.size_bytes()) {
    cudaFree(mPtr);
    cudaMalloc(&mPtr, rhs.size_bytes());
    mNBytes = rhs.size_bytes();
  }
  cudaMemcpy(mPtr, rhs.data(), mNBytes, cudaMemcpyDeviceToDevice);
  return *this;
}

bool GPUdata::operator==(const GPUdata &rhs) {
  if (mNBytes != rhs.size_bytes()) return false;
  uint8_t *left = new uint8_t[mNBytes];
  uint8_t *right = new uint8_t[mNBytes];
  cudaMemcpy(left, mPtr, mNBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(right, rhs.data(), mNBytes, cudaMemcpyDeviceToHost);
  int cmp = memcmp(left, right, mNBytes);
  delete[] left;
  delete[] right;
  return cmp == 0;
}

bool GPUdata::operator!=(const GPUdata &rhs) {
  return !(*this == rhs);
}

void GPUdata::resize(uint64_t size) {
  if (size == mNBytes) return;
  uint8_t *newData;
  cudaError_t err = cudaMalloc(&newData, size);
  check_call("GPUdata::resize\n");
  if (mPtr != nullptr) {
    cudaMemcpy(newData, mPtr, std::min(size, mNBytes), cudaMemcpyDeviceToDevice);
    cudaFree(mPtr);
  }
  mPtr = newData;
  mNBytes = size;
}

void GPUdata::load(const uint8_t *data) {
  cudaMemcpy(mPtr, data, mNBytes, cudaMemcpyDeviceToDevice);
}

void GPUdata::load(const char *filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  char *buffer = new char[mNBytes];
  ifs.read(buffer, mNBytes);
  cudaMemcpy(mPtr, buffer, mNBytes, cudaMemcpyHostToDevice);
  ifs.close();
  delete[] buffer;
}

void GPUdata::save(const char *filename) {
  std::ofstream ofs(filename, std::ios::out | std::ios::app | std::ios::binary);
  char *buffer = new char[mNBytes];
  cudaMemcpy(buffer, mPtr, mNBytes, cudaMemcpyDeviceToHost);
  ofs.write(buffer, mNBytes);
  ofs.close();
  delete[] buffer;
}

void GPUdata::clear() {
  cudaMemset(mPtr, 0, mNBytes);
}

void GPUdata::xor_d(GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  check_call("GPUdata::xor_d\n");
}

std::ostream& operator<<(std::ostream &os, GPUdata &obj) {
  uint8_t *tmp = new uint8_t[obj.size_bytes()];
  cudaMemcpy(tmp, obj.data(), obj.size_bytes(), cudaMemcpyDeviceToHost);
  for (uint64_t i = 0; i < obj.size_bytes(); i += 16 * sizeof(OTblock)) {
    for (uint64_t j = 0; j < 16 * sizeof(OTblock); j += sizeof(OTblock)) {
      os << std::hex << std::setw(2) << std::setfill('0') << int(tmp[i+j]) << " ";
    }
    os << std::endl;
  }
  os << std::endl;
  delete[] tmp;
  return os;
}
