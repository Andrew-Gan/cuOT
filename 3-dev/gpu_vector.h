#ifndef __GPU_VECTOR_H__
#define __GPU_VECTOR_H__

#include "gpu_matrix.h"
#include "gpu_ops.h"

template<typename T>
class GPUvector : public GPUmatrix<T> {
public:
  using GPUdata::xor_async;

  GPUvector(uint64_t len) : GPUmatrix<T>(1, len) {}
  uint64_t size() { return this->mCols; }
  void set(uint64_t i, T &val) { GPUmatrix<T>::set(0, i, val); }
  void resize(uint64_t len) { GPUmatrix<T>::resize(1, len); }
  void sum_async(uint64_t nPartition, uint64_t blkPerPart, cudaStream_t s);
  void xor_async(GPUvector<T> &rhs, uint64_t offs, cudaStream_t s);
};

// nPartition: number of partitions to reduce to separate totals
// blkPerPart: number of OTblocks in a partition
template<typename T>
void GPUvector<T>::sum_async(uint64_t nPartition, uint64_t blkPerPart, cudaStream_t s) {
  uint64_t blockSize, nBlocks, mem, u64PerPartition;

  uint8_t *buffer;
  cudaMallocAsync(&buffer, this->mNBytes / std::min((uint64_t) 1024, blkPerPart), s);

  uint64_t *in = (uint64_t*) buffer;
  uint64_t *out = (uint64_t*) this->mPtr;

  for (uint64_t remBlocks = blkPerPart; remBlocks > 1; remBlocks /= 1024) {
    std::swap(in, out);

    u64PerPartition = 2 * remBlocks;
    blockSize = u64PerPartition >= 2048 ? 1024 : u64PerPartition / 2;
    nBlocks = nPartition * u64PerPartition / (2 * blockSize);
    mem = blockSize * sizeof(uint64_t);
    xor_reduce_gpu<<<nBlocks, blockSize, mem, s>>>(out, in);
  }

  if (out != (uint64_t*) this->mPtr) {
    cudaMemcpyAsync(this->mPtr, out, nPartition * sizeof(T), cudaMemcpyDeviceToDevice, s);
  }

  cudaFreeAsync(buffer, s);
}

template<typename T>
void GPUvector<T>::xor_async(GPUvector<T> &rhs, uint64_t offs, cudaStream_t s) {
  uint64_t min = std::min(this->mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  xor_gpu<<<nBlock, 1024, 0, s>>>(this->mPtr, (uint8_t*) (rhs.data() + offs), min);
}

#endif
