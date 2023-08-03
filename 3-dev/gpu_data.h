#ifndef __GPU_DATA_H__
#define __GPU_DATA_H__

#include "util.h"

class GPUdata {
public:
  GPUdata(uint64_t n);
  GPUdata(const GPUdata &blk);
  virtual ~GPUdata();
  GPUdata& operator&=(const GPUdata &rhs);
  GPUdata& operator^=(const GPUdata &rhs);
  GPUdata& operator=(const GPUdata &rhs);
  bool operator==(const GPUdata &rhs);
  bool operator!=(const GPUdata &rhs);
  uint8_t* data() const { return mPtr; }
  uint64_t size_bytes() const { return mNBytes; }
  void resize(uint64_t size);
  void load(const uint8_t *data);
  void load(const char *filename);
  void save(const char *filename);
  void clear();
  void xor_async(GPUdata &rhs, cudaStream_t s);
  void copy_async(GPUdata &rhs, cudaStream_t s);

protected:
  uint8_t *mPtr = nullptr;
  uint64_t mNBytes = 0;
};

#endif
