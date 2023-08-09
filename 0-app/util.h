#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include "event_log.h"

#define AES_KEYLEN 16
#define AES_BSIZE 256
#define AES_PADDING (AES_BSIZE / 4 * 16)

#define CUDA_CALL(e) if (e != cudaSuccess) \
  fprintf(stderr, "%s\n", cudaGetErrorString(e));

enum Role { Sender, Recver };

struct AES_ctx {
  uint8_t roundKey[176];
};

struct OTblock {
  uint32_t data[4];
};

union UByte4 {
  unsigned int uival;
  unsigned char ubval[4];
};

#endif
