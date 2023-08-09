#include <iomanip>
#include <mutex>

#include "gpu_data.h"
#include "gpu_ops.h"

GPUdata::GPUdata(uint64_t n) : mNBytes(n) {
  CUDA_CALL(cudaMalloc(&mPtr, n));
}

GPUdata::GPUdata(const GPUdata &blk) : GPUdata(blk.size_bytes()) {
  CUDA_CALL(cudaMemcpy(mPtr, blk.data(), mNBytes, cudaMemcpyDeviceToDevice));
}

GPUdata::~GPUdata() {
  if (mPtr != nullptr) CUDA_CALL(cudaFree(mPtr));
}

GPUdata& GPUdata::operator&=(const GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  and_gpu<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  cudaDeviceSynchronize();
  return *this;
}

GPUdata& GPUdata::operator^=(const GPUdata &rhs) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  xor_gpu<<<nBlock, 1024>>>(mPtr, rhs.data(), min);
  cudaDeviceSynchronize();
  return *this;
}

GPUdata& GPUdata::operator=(const GPUdata &rhs) {
  if (mNBytes != rhs.size_bytes()) resize(rhs.size_bytes());
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
  CUDA_CALL(cudaMalloc(&newData, size));
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
  std::ofstream ofs(filename, std::ios::out | std::ios::binary);
  char *buffer = new char[mNBytes];
  cudaMemcpy(buffer, mPtr, mNBytes, cudaMemcpyDeviceToHost);
  ofs.write(buffer, mNBytes);
  ofs.close();
  delete[] buffer;
}

void GPUdata::clear() {
  cudaMemset(mPtr, 0, mNBytes);
}

void GPUdata::xor_async(GPUdata &rhs, cudaStream_t s) {
  uint64_t min = std::min(mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  xor_gpu<<<nBlock, 1024, 0, s>>>(mPtr, rhs.data(), min);
}

void GPUdata::copy_async(GPUdata &rhs, cudaStream_t s) {
  if (mNBytes != rhs.size_bytes()) {
    CUDA_CALL(cudaFree(mPtr));
    CUDA_CALL(cudaMallocAsync(&mPtr, rhs.size_bytes(), s));
    mNBytes = rhs.size_bytes();
  }
  cudaMemcpyAsync(mPtr, rhs.data(), mNBytes, cudaMemcpyDeviceToDevice, s);
}
