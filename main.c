#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "aes.h"
#include "aesni.h"
#include "aesgpu.h"

#include "pprf.h"

#include "utilsBox.h"

size_t _get_filesize(FILE *fp) {
  size_t fpos = ftell(fp);
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, fpos, SEEK_SET);
  return fsize;
}

void _tester(
  void (*initialiser)(), void (*encryptor) (), void (*decryptor) (),
  const uint8_t *key, AES_buffer *buf, int nThreads, const char *msg) {

  uint8_t *original = malloc(buf->length * sizeof(*original) + 1024);
  memcpy(original, buf->content, buf->length);

  AES_ctx ctx;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  initialiser(&ctx, key);
  encryptor(&ctx, buf, nThreads);
  // assert(memcmp(original, buf, 32) != 0);
  decryptor(&ctx, buf, nThreads);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = end.tv_sec - start.tv_sec;
  elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  assert(memcmp(original, buf->content, buf->length) == 0);
  free(original);

  printf("AES enc and dec using %s: %0.4f s\n", msg, elapsed);
}

int main(int argc, char** argv) {
  if (argc < 2 || (strcmp(argv[1], "enc") && strcmp(argv[1], "exp"))) {
    fprintf(stderr, "Usage: ./aes mode[enc|exp] ...");
    return EXIT_FAILURE;
  }

  if (strcmp(argv[1], "enc") == 0) {
    if (argc < 6) {
      fprintf(stderr, "Usage: ./aes enc input key textLen nThreads\n");
      return EXIT_FAILURE;
    }
    FILE *inputFile = fopen(argv[2], "rb");
    FILE *keyFile = fopen(argv[3], "rb");
    size_t dataSize = atoi(argv[4]);
    if (inputFile == NULL || keyFile == NULL) {
      fprintf(stderr, "Error: Either files does not exist\n");
      return EXIT_FAILURE;
    }
    if (_get_filesize(keyFile) < AES_KEYLEN) {
      fprintf(stderr, "Error: Key file is less than 128 bits in length\n");
      return EXIT_FAILURE;
    }

    AES_buffer input;
    input.length = dataSize;
    // round input length to next multiple of block length
    if (input.length % AES_BLOCKLEN != 0) {
      input.length = AES_BLOCKLEN * (1 + input.length / AES_BLOCKLEN);
    }
    int nThreads = atoi(argv[5]);
    printf("inputSize: %lu, nThreads: %d\n", input.length, nThreads);

    input.content = malloc(input.length * sizeof(*(input.content)) + 1024);
    fread(input.content, sizeof(*(input.content)), input.length, inputFile);
    uint8_t *key = malloc(AES_KEYLEN * sizeof(*key));
    fread(key, sizeof(*key), AES_KEYLEN, keyFile);

    _tester(aes_init_ctx, aes_ecb_encrypt, aes_ecb_decrypt, key, &input, nThreads, "AES");
    _tester(aesni_init_ctx, aesni_ecb_encrypt, aesni_ecb_decrypt, key, &input, nThreads, "AESNI");
    _tester(aes_init_ctx, aesgpu_ecb_encrypt, aesgpu_ecb_decrypt, key, &input, nThreads, "AESGPU");

    free(input.content);
    free(key);
    fclose(inputFile);
    fclose(keyFile);
  }
  else if (strcmp(argv[1], "exp") == 0) {
    if (argc < 3) {
      fprintf(stderr, "Usage: ./aes exp depth nThread\n");
    }
    size_t depth = atoi(argv[2]);
    size_t numNodes = pow(2, depth + 1) - 1;
    AES_block *blocks = malloc(sizeof(*blocks) * numNodes);
    blocks[0] = (AES_block){ .data[0] = 123546, .data[1] = 789012 };
    size_t nThread = atoi(argv[3]);

    test_expand(blocks, depth, aesni_init_ctx, aesni_ecb_encrypt, nThread, "AESNI");

    // for (int i = 0; i < numNodes; i++) {
    //   printf("0x%x%x ", blocks[i].data[0], blocks[i].data[1]);
    // }
    // printf("\n");

    free(blocks);
  }

  return EXIT_SUCCESS;
}
