#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include "event_log.h"

#define AES_KEYLEN 16
#define BLK_SIZE 16
#define AES_BSIZE 256

typedef struct {
  uint8_t roundKey[176];
} AES_ctx;

typedef struct {
  uint32_t data[BLK_SIZE / 4];
} TreeNode;

typedef struct {
  uint64_t nBits = 0;
  uint64_t *nonZeros;
  uint64_t weight = 0;
} SparseVector;

typedef struct {
  uint64_t rows, cols;
  uint8_t *data;
} Matrix;

union UByte4 {
  unsigned int uival;
  unsigned char ubval[4];
};

#endif
