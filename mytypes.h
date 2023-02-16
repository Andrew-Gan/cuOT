#ifndef __MYTYPES_H__
#define __MYTYPES_H__

#define AES_BLOCKLEN 16
#define AES_KEYLEN 16 // Key length in bytes
#define AES_keyExpSize 176
#define NUM_ROUNDS 10
#define NUM_CPU_THREAD 16

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
  uint64_t data[2];
} AES_block;

typedef struct {
  void (*encryptor)();
  AES_block *tree;
  size_t idx;
} ThreadTreeArgs;

#endif
