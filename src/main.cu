#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "aes.h"
#include "aesni.h"
#include "pprf_cpu.h"
#include "pprf_gpu.h"
#include <math.h>


#include "utilsBox.h"

size_t _get_filesize(FILE *fp) {
  size_t fpos = ftell(fp);
  fseek(fp, 0, SEEK_END);
  size_t fsize = ftell(fp);
  fseek(fp, fpos, SEEK_SET);
  return fsize;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./aes depth numTrees\n");
    return EXIT_FAILURE;
  }
  size_t depth = atoi(argv[1]);
  int numTrees = atoi(argv[2]);
  size_t numNodes = pow(2, depth + 1) - 1;
  size_t numLeaves = numNodes / 2 + 1;
  TreeNode *tree = (TreeNode*) malloc(sizeof(*tree) * numNodes);
  tree[0].data[0] = 123456;
  tree[0].data[1] = 7890123;

  TreeNode *tree2 = (TreeNode*) malloc(sizeof(*tree) * numNodes);
  memcpy(tree2, tree, sizeof(*tree) * numNodes);

  TreeNode *leaves;
  cudaMallocHost(&leaves, sizeof(*leaves) * numLeaves);

  printf("Depth: %lu, Nodes: %lu, Threads: %d\n", depth, numNodes, numTrees);
  aescpu_tree_expand(tree, depth, aes_init_ctx, aes_ecb_encrypt, "AES", numTrees);
  aescpu_tree_expand(tree2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI", numTrees);
  aesgpu_tree_expand(tree, leaves, depth);

  assert(memcmp(&tree[numLeaves - 1], &tree2[numLeaves - 1], sizeof(*tree) * numLeaves) == 0);
  assert(memcmp(&tree2[numLeaves - 1], leaves, sizeof(*tree) * numLeaves) == 0);

  free(tree);
  free(tree2);
  cudaFreeHost(leaves);

  return EXIT_SUCCESS;
}
