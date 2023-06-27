#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include "event_log.h"

#define AES_KEYLEN 16
#define BLK_SIZE 16
#define AES_BSIZE 256

struct AES_ctx {
  uint8_t roundKey[176];
};

struct OTBlock {
  uint32_t data[BLK_SIZE / 4];
};

struct SparseVector {
  uint64_t nBits = 0;
  uint64_t *nonZeros;
  uint64_t weight = 0;
};

union UByte4 {
  unsigned int uival;
  unsigned char ubval[4];
};

#endif
