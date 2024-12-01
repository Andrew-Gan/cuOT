#ifndef __GPU_DATA_H__
#define __GPU_DATA_H__

#include <iostream>

class GPUdata {
public:
  GPUdata() {}
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
  bool resize(uint64_t size);
  void read_from_cpu(const void *data, uint64_t size = 0);
  void write_to_cpu(void *data, uint64_t size = 0, uint64_t start = 0);
  void read_from_gpu(const void *data, uint64_t size = 0, int gpu = -1);
  void write_to_gpu(void *data, uint64_t size = 0, uint64_t start = 0, int gpu = -1);
  void load(const char *filename);
  void save(const char *filename);
  void clear();
  void xor_d(GPUdata &rhs);
  void get_mem_handle(uint8_t *handleBytes);

protected:
  int mDevice = 0;
  uint8_t *mPtr = nullptr;
  uint64_t mNBytes = 0;
  uint64_t mAllocated = 0;
};

std::ostream& operator<<(std::ostream &os, GPUdata &obj);

#endif
