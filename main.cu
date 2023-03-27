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
  void (*encryptor) (AES_ctx*, AES_buffer*, int),
  void (*decryptor) (AES_ctx*, AES_buffer*, int),
  const uint8_t *key, AES_buffer *buf, const char *msg, int numThread) {

  uint8_t *original = (uint8_t*) malloc(buf->length * sizeof(*original) + 1024);
  memcpy(original, buf->content, buf->length);

  AES_ctx ctx;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  initialiser(&ctx, key);
  encryptor(&ctx, buf, numThread);
  assert(memcmp(original, buf, 128) != 0);
  decryptor(&ctx, buf, numThread);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000;
  elapsed += (end.tv_nsec - start.tv_nsec) / 1000000.0;

  assert(memcmp(original, buf->content, buf->length) == 0);
  free(original);

  printf("AES enc and dec using %s: %0.4f ms\n", msg, elapsed);
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
    int numThread = atoi(argv[5]);
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

    test_xcrypt(aes_init_ctx, aes_ecb_encrypt, aes_ecb_decrypt, key, &input, "AES", numThread);
    test_xcrypt(aesni_init_ctx, aesni_ecb_encrypt, aesni_ecb_decrypt, key, &input, "AESNI", numThread);
    test_xcrypt(aes_init_ctx, aesgpu_ecb_encrypt, aesgpu_ecb_decrypt, key, &input, "AESGPU", numThread);

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
    int numThread = atoi(argv[3]);
    size_t numNodes = pow(2, depth + 1) - 1;
    size_t numLeaves = numNodes / 2 + 1;
    TreeNode *tree = (TreeNode*) malloc(sizeof(*tree) * numNodes);
    tree[0].data[0] = 123456;
    tree[0].data[1] = 7890123;

    TreeNode *tree2 = (TreeNode*) malloc(sizeof(*tree) * numNodes);
    memcpy(tree2, tree, sizeof(*tree) * numNodes);

    TreeNode *leaves;
    cudaMallocHost(&leaves, sizeof(*leaves) * numLeaves);

    printf("Depth: %lu, Nodes: %lu, Threads: %d\n", depth, numNodes, numThread);
    aescpu_tree_expand(tree, depth, aes_init_ctx, aes_ecb_encrypt, "AES", numThread);
    aescpu_tree_expand(tree2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI", numThread);
    aesgpu_tree_expand(tree, leaves, depth);

    assert(memcmp(&tree[numLeaves - 1], &tree2[numLeaves - 1], sizeof(*tree) * numLeaves) == 0);
    assert(memcmp(&tree2[numLeaves - 1], leaves, sizeof(*tree) * numLeaves) == 0);

    free(tree);
    free(tree2);
    cudaFreeHost(leaves);
  }

  return EXIT_SUCCESS;
}
