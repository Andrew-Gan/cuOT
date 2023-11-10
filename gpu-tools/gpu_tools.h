#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include "gpu_data.h"
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "event_log.h"

#define AES_KEYLEN 16
#define AES_PADDING (AES_BSIZE / 4 * 16)

#define cuda_init() cudaFree(0)

inline void CUDA_CALL(cudaError_t err) {
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    assert(err == cudaSuccess);
}

enum Role { Sender, Recver };

struct AES_ctx {
  uint8_t roundKey[176];
};

struct OTblock {
  uint32_t data[4];
};

// aliases
using blk = OTblock;
using vec = GPUvector<blk>;
using mat = GPUmatrix<blk>;

#endif
