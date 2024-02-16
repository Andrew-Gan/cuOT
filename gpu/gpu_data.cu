#include <iomanip>
#include <fstream>
#include <mutex>

#include "gpu_data.h"
#include "gpu_ops.h"
#include "gpu_tests.h"

GPUdata::GPUdata(uint64_t n) : mNBytes(n) {
  cudaError_t err = cudaMalloc(&mPtr, n);
  cudaDeviceSynchronize();
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
  cudaDeviceSynchronize();
  return *this;
}

GPUdata& GPUdata::operator^=(const GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  cudaDeviceSynchronize();
  return *this;
}

GPUdata& GPUdata::operator=(const GPUdata &rhs) {
  if (mNBytes != rhs.size_bytes()) {
    cudaFreeAsync(mPtr, 0);
    cudaMalloc(&mPtr, rhs.size_bytes());
    mNBytes = rhs.size_bytes();
  }
  cudaMemcpyAsync(mPtr, rhs.data(), mNBytes, cudaMemcpyDeviceToDevice);
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
  if (mPtr == nullptr)
    cudaMalloc(&mPtr, size);
  else {
    uint8_t *oldData = mPtr;
    cudaMalloc(&mPtr, size);
    cudaMemcpyAsync(mPtr, oldData, std::min(size, mNBytes), cudaMemcpyDeviceToDevice);
    cudaFreeAsync(oldData, 0);
  }
  mNBytes = size;
}

void GPUdata::load(const void *data, uint64_t size) {
  uint64_t cpy = size == 0 ? mNBytes : size;
  cudaMemcpyAsync(mPtr, data, cpy, cudaMemcpyDeviceToDevice);
}

void GPUdata::load(const char *filename) {
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  char *buffer = new char[mNBytes];
  ifs.read(buffer, mNBytes);
  cudaMemcpyAsync(mPtr, buffer, mNBytes, cudaMemcpyHostToDevice);
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
}

std::ostream& operator<<(std::ostream &os, GPUdata &obj) {
  blk *tmp = new blk[obj.size_bytes() / sizeof(blk)];
  cudaMemcpy(tmp, obj.data(), obj.size_bytes(), cudaMemcpyDeviceToHost);
  for (uint64_t i = 0; i < obj.size_bytes() / sizeof(blk); i += 16) {
    for (uint64_t j = 0; j < 16; j++) {
      os << std::hex << std::setw(2) << std::setfill('0') << int(((uint8_t*)(tmp+i+j))[0]) << " ";
    }
    os << std::endl;
  }
  os << std::dec;
  delete[] tmp;
  return os;
}
