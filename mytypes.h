#ifndef __MYTYPES_H__
#define __MYTYPES_H__

#define AES_BLOCKLEN 16
#define AES_KEYLEN 16 // Key length in bytes
#define AES_keyExpSize 176
#define NUM_ROUNDS 10

#define TREENODE_SIZE (2 * AES_BLOCKLEN)

#define NUM_CPU_THREAD 16
#define NUM_SAMPLES 16

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <wmmintrin.h>

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
  void (*encryptor)(AES_ctx*, AES_buffer*);
  TreeNode *tree;
  size_t idx;
} ThreadTreeArgs;

#endif
