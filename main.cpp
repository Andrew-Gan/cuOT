#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "aes.h"
#include "aesni.h"
#include "aesgpu.h"
#include <math.h>

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
  void (*initialiser)(AES_ctx*, const uint8_t*),
  void (*encryptor) (AES_ctx*, AES_buffer*),
  void (*decryptor) (AES_ctx*, AES_buffer*),
  const uint8_t *key, AES_buffer *buf, const char *msg) {

  uint8_t *original = (uint8_t*) malloc(buf->length * sizeof(*original) + 1024);
  memcpy(original, buf->content, buf->length);

  AES_ctx ctx;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  initialiser(&ctx, key);
  encryptor(&ctx, buf);
  assert(memcmp(original, buf, 128) != 0);
  decryptor(&ctx, buf);
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
    if (argc < 5) {
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
    printf("inputSize: %lu\n", input.length);

    input.content = (uint8_t*) malloc(input.length * sizeof(*(input.content)) + 1024);
    fread(input.content, sizeof(*(input.content)), input.length, inputFile);
    uint8_t *key = (uint8_t*) malloc(AES_KEYLEN * sizeof(*key));
    fread(key, sizeof(*key), AES_KEYLEN, keyFile);

    test_xcrypt(aes_init_ctx, aes_ecb_encrypt, aes_ecb_decrypt, key, &input, "AES");
    test_xcrypt(aesni_init_ctx, aesni_ecb_encrypt, aesni_ecb_decrypt, key, &input, "AESNI");
    test_xcrypt(aes_init_ctx, aesgpu_ecb_encrypt, aesgpu_ecb_decrypt, key, &input, "AESGPU");

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
    AES_block *blocks = (AES_block*) malloc(sizeof(*blocks) * numNodes);
    blocks[0].data[0] = 123456;
    blocks[1].data[1] = 7890123;

    AES_block *blocks2 = (AES_block*) malloc(sizeof(*blocks) * numNodes);
    memcpy(blocks2, blocks, sizeof(*blocks) * numNodes);

    AES_block *blocks3 = (AES_block*) malloc(sizeof(*blocks) * numNodes);
    memcpy(blocks3, blocks, sizeof(*blocks) * numNodes);

    printf("Depth: %lu, Nodes: %lu\n", depth, numNodes);
    aescpu_tree_expand(blocks, depth, aes_init_ctx, aes_ecb_encrypt, "AES");
    aescpu_tree_expand(blocks2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI");
    aesgpu_tree_expand(blocks3, depth);

    // print_tree(blocks2, depth);
    // print_tree(blocks3, depth);

    assert(memcmp(blocks, blocks2, sizeof(*blocks) * numNodes) == 0);
    assert(memcmp(blocks2, blocks3, sizeof(*blocks) * numNodes) == 0);

    free(blocks);
    free(blocks2);
    free(blocks3);
  }

  return EXIT_SUCCESS;
}
