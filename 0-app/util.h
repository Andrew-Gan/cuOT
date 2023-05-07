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
#define NUM_SAMPLES 8
#define CHUNK_SIDE (1<<17)

#define EXP_NUM_THREAD 16
#define AES_BSIZE 256
// #define UNIT_TEST

typedef struct {
  uint8_t roundKey[176];
} AES_ctx;

typedef struct {
  uint32_t data[TREENODE_SIZE / 4];
} TreeNode;

typedef struct {
  size_t n; // num bits
  uint8_t *data;
} Vector;

typedef struct {
  size_t rows, cols;
  uint8_t *data;
} Matrix;

union UByte4 {
  unsigned int uival;
  unsigned char ubval[4];
};

__global__
void xor_gpu(Vector c, Vector a, Vector b);

__global__
void and_gpu(Vector c, Vector a, uint8_t b);

__global__
void cmp_gpu(bool *c, uint8_t *a, uint8_t *b);

__global__
void print_gpu(uint8_t *a, size_t n);

#endif
