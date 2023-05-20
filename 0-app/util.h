#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdint.h>
#include <iostream>
#include "event_log.h"

#define AES_BLOCKLEN 16
#define AES_KEYLEN 16 // Key length in bytes
#define AES_keyExpSize 176
#define NUM_ROUNDS 10
#define PADDED_LEN 1024

#define TREENODE_SIZE AES_BLOCKLEN
#define NUM_SAMPLES 8
#define CHUNK_SIDE (1<<17)

#define EXP_NUM_THREAD 16
#define AES_BSIZE 256

enum Role { Sender, Recver };

typedef struct {
  uint8_t roundKey[176];
} AES_ctx;

typedef struct {
  uint32_t data[TREENODE_SIZE / 4];
} TreeNode;

typedef struct {
  size_t nBits = 0;
  uint8_t *data = nullptr;
} Vector;

typedef struct {
  size_t nBits = 0;
  size_t *nonZeros;
  size_t weight = 0;
} SparseVector;

typedef struct {
  size_t rows, cols;
  uint8_t *data;
} Matrix;

union UByte4 {
  unsigned int uival;
  unsigned char ubval[4];
};

#endif
