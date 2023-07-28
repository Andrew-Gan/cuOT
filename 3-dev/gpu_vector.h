#ifndef __GPU_VECTOR_H__
#define __GPU_VECTOR_H__

#include "gpu_matrix.h"
#include "gpu_ops.h"

template<typename T>
class GPUvector : public GPUmatrix<T> {
public:
  GPUvector(uint64_t len) : GPUmatrix<T>(1, len) {}
  uint64_t size() { return this->mCols; }
  void set(uint64_t i, T &val) { GPUmatrix<T>::set(0, i, val); }
  void resize(uint64_t len) { GPUmatrix<T>::resize(1, len); }
  void sum_async(uint64_t nPartition, uint64_t blkPerPart, cudaStream_t s);
};

// nPartition: number of partitions to reduce to separate totals
// blkPerPart: number of OTblocks in a partition
template<typename T>
void GPUvector<T>::sum_async(uint64_t nPartition, uint64_t blkPerPart, cudaStream_t s) {
  uint64_t blockSize, blockSize2, nBlocks, mem, u64PerPartition;
  for (uint64_t remBlocks = blkPerPart; remBlocks > 1; remBlocks /= 1024) {
    u64PerPartition = 2 * remBlocks;
    blockSize = u64PerPartition >= 2048 ? 1024 : u64PerPartition / 2;
    nBlocks = nPartition * u64PerPartition / (2 * blockSize);
    mem = blockSize * sizeof(uint64_t);
    xor_reduce_gpu<<<nBlocks, blockSize, mem, s>>>((uint64_t*) this->mPtr);
    cudaDeviceSynchronize();

    blockSize2 = std::min(1024lu, nBlocks);
    nBlocks = nBlocks < 1024 ? 1 :( nBlocks + 1023) / 1024;
    xor_reduce_packer_gpu<<<nBlocks, blockSize2, 0, s>>>((uint64_t*) this->mPtr, 2 * blockSize);
    cudaDeviceSynchronize();
  }

}

#endif
