#ifndef __MYTYPES_H__
#define __MYTYPES_H__

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <wmmintrin.h>

#define AES_BLOCKLEN 16
#define AES_KEYLEN 16 // Key length in bytes
#define AES_keyExpSize 176
#define NUM_ROUNDS 10
#define PADDED_LEN 1024

#define TREENODE_SIZE AES_BLOCKLEN
#define NUM_SAMPLES 16
#define CHUNK_SIDE 32768

typedef struct {
  uint8_t roundKey[320];
} AES_ctx;

typedef struct {
  size_t length;
  uint8_t *content;
} AES_buffer;

typedef struct {
  AES_ctx *ctx;
  AES_buffer *buf;
  size_t start;
  size_t end;
} ThreadArgs;

typedef struct {
  uint32_t data[TREENODE_SIZE / 4];
} TreeNode;

typedef struct {
  size_t n;
  uint8_t *data;
} Vector;

typedef struct {
  size_t rows, cols;
  uint8_t *data;
} Matrix;

#endif
