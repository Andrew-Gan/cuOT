#include <iomanip>
#include <fstream>
#include <mutex>

#include "gpu_data.h"
#include "gpu_ops.h"
#include "gpu_tests.h"

GPUdata::GPUdata(uint64_t n) : mNBytes(n), mAllocated(n) {
  cudaGetDevice(&mDevice);
  cudaError_t err = cudaMalloc(&mPtr, n);
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

GPUdata::GPUdata(const GPUdata &blk) : GPUdata(blk.size_bytes()) {
  cudaGetDevice(&mDevice);
  cudaError_t err = cudaMemcpyPeer(mPtr, mDevice, blk.data(), blk.mDevice, mNBytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

GPUdata::~GPUdata() {
  if (mPtr != nullptr) {
    cudaSetDevice(mDevice);
    cudaError_t err = cudaFree(mPtr);
    if (err != cudaSuccess) {
      printf("Failed to free gpu mem: %s\n", cudaGetErrorString(err));
    }
  }
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
  if (mAllocated < rhs.size_bytes()) {
    cudaFree(mPtr);
    cudaError_t err = cudaMalloc(&mPtr, rhs.size_bytes());
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    mAllocated = rhs.size_bytes();
  }
  mNBytes = rhs.size_bytes();
  cudaMemcpyPeer(mPtr, mDevice, rhs.data(), rhs.mDevice, mNBytes);
  return *this;
}

bool GPUdata::operator==(const GPUdata &rhs) {
  if (mNBytes != rhs.size_bytes()) return false;
  uint8_t *left = new uint8_t[mNBytes];
  uint8_t *right = new uint8_t[mNBytes];
  cudaMemcpy(left, mPtr, mNBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(right, rhs.data(), mNBytes, cudaMemcpyDeviceToHost);
  int cmp = memcmp(left, right, mNBytes);

  if (cmp != 0) {
    std::cout << "First inequality at: ";
    for (uint64_t i = 0; i < mNBytes; i += 16) {
      if (left[i] != right[i]) {
        std::cout << i / 16 << std::endl;;
        break;
      }
    }
  }

  delete[] left;
  delete[] right;
  return cmp == 0;
}

bool GPUdata::operator!=(const GPUdata &rhs) {
  return !(*this == rhs);
}

void GPUdata::resize(uint64_t size) {
  cudaGetDevice(&mDevice);
  if (size == mNBytes)
    return;
  if (size == 0) {
    cudaFree(mPtr);
    mAllocated = size;
  }

  if (mAllocated == 0) {
    cudaError_t err = cudaMalloc(&mPtr, size);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    mAllocated = size;
  }
  else if (size > mAllocated) {
    cudaFree(mPtr);
    cudaError_t err = cudaMalloc(&mPtr, size);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    mAllocated = size;
  }
  mNBytes = size;
}

void GPUdata::load(const void *data, uint64_t size) {
  uint64_t cpy = size == 0 ? mNBytes : size;
  cudaMemcpy(mPtr, data, cpy, cudaMemcpyDeviceToDevice);
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
