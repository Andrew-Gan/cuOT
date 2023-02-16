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

static void test_xcrypt(
  void (*initialiser)(), void (*encryptor) (), void (*decryptor) (),
  const uint8_t *key, AES_buffer *buf, int nThreads, const char *msg) {

  uint8_t *original = malloc(buf->length * sizeof(*original) + 1024);
  memcpy(original, buf->content, buf->length);

  AES_ctx ctx;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  initialiser(&ctx, key);
  encryptor(&ctx, buf, nThreads);
  assert(memcmp(original, buf, 128) != 0);
  decryptor(&ctx, buf, nThreads);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000;
  elapsed += (end.tv_nsec - start.tv_nsec) / 1000000.0;

  assert(memcmp(original, buf->content, buf->length) == 0);
  free(original);

  printf("AES enc and dec using %s: %0.4f ms\n", msg, elapsed);
}

static void print_tree(AES_block *tree, size_t depth) {
  size_t startingIdx = 0;
  size_t width = 1;
  for (size_t d = 0; d <= depth; d++) {
    printf("Depth: %d\n", d);
    for (size_t idx = startingIdx; idx < startingIdx + width; idx++) {
      printf("0x%x%x ", tree[idx].data[0], tree[idx].data[1]);
    }
    printf("\n");
    startingIdx += width;
    width *= 2;
  }
  printf("\n");
}

int main(int argc, char** argv) {
  if (argc < 2 || (strcmp(argv[1], "enc") && strcmp(argv[1], "exp"))) {
    fprintf(stderr, "Usage: ./aes mode[enc|exp] ...\n");
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
      fprintf(stderr, "Error: Either file does not exist\n");
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

    test_xcrypt(aes_init_ctx, aes_ecb_encrypt, aes_ecb_decrypt, key, &input, nThreads, "AES");
    test_xcrypt(aesni_init_ctx, aesni_ecb_encrypt, aesni_ecb_decrypt, key, &input, nThreads, "AESNI");
    test_xcrypt(aes_init_ctx, aesgpu_ecb_encrypt, aesgpu_ecb_decrypt, key, &input, nThreads, "AESGPU");

    free(input.content);
    free(key);
    fclose(inputFile);
    fclose(keyFile);
  }
  else if (strcmp(argv[1], "exp") == 0) {
    if (argc < 3) {
      fprintf(stderr, "Usage: ./aes exp depth\n");
      return EXIT_FAILURE;
    }
    size_t depth = atoi(argv[2]);
    size_t numNodes = pow(2, depth + 1) - 1;
    AES_block *blocks = malloc(sizeof(*blocks) * numNodes);
    blocks[0] = (AES_block){ .data[0] = 123546, .data[1] = 789012 };

    AES_block *blocks2 = malloc(sizeof(*blocks) * numNodes);
    memcpy(blocks2, blocks, sizeof(*blocks) * numNodes);

    AES_block *blocks3 = malloc(sizeof(*blocks) * numNodes);
    memcpy(blocks3, blocks, sizeof(*blocks) * numNodes);

    printf("Depth: %lu, Nodes: %lu\n", depth, numNodes);
    // test_expand(blocks, depth, aes_init_ctx, aes_ecb_encrypt, "AES");
    test_expand(blocks2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI");
    test_expand(blocks3, depth, aes_init_ctx, aesgpu_ecb_encrypt, "AESGPU");

    // assert(memcmp(blocks, blocks2, sizeof(*blocks) * numNodes) == 0);
    assert(memcmp(blocks2, blocks3, sizeof(*blocks) * numNodes) == 0);

    free(blocks);
    free(blocks2);
    free(blocks3);
  }

  return EXIT_SUCCESS;
}
