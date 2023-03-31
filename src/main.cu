#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <thread>

#include "aes.h"
#include "aesni.h"
#include "pprf_cpu.h"
#include "pprf_gpu.h"
#include "ldpc.h"
#include "gemm_gpu.h"

#include "utilsBox.h"

void print_leaves(TreeNode *leaves, int numLeaves) {
  for(int i = 0; i < numLeaves; i++) {
    for(int j = 0; j < TREENODE_SIZE / 4; j++) {
      printf("0x%x ", leaves[i].data[j]);
    }
  }
  printf("\n");
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./pprf depth numTrees\n");
    return EXIT_FAILURE;
  }

  size_t depth = atoi(argv[1]);
  int numTrees = atoi(argv[2]);
  size_t numNodes = pow(2, depth + 1) - 1;
  size_t numLeaves = numNodes / 2 + 1;
  TreeNode *root = (TreeNode*) malloc(sizeof(*root));
  root->data[0] = 123456;
  root->data[1] = 7890123;

  // TreeNode *tree2 = (TreeNode*) malloc(sizeof(*tree) * numNodes);
  // memcpy(tree2, tree, sizeof(*tree) * numNodes);

  printf("Depth: %lu, Nodes: %lu, Threads: %d\n", depth, numNodes, numTrees);
  // aescpu_tree_expand(tree, depth, aes_init_ctx, aes_ecb_encrypt, "AES", numTrees);
  // aescpu_tree_expand(tree2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI", numTrees);

  TreeNode *d_multiPprf;
  cudaMalloc(&d_multiPprf, sizeof(*d_multiPprf) * numLeaves);
  int *nonZeroRows = (int*) malloc(numTrees * sizeof(*nonZeroRows));
  std::thread senderExp(pprf_sender_gpu, root, depth, numTrees);
  std::thread recverExp(pprf_recver_gpu, d_multiPprf, nonZeroRows, depth, numTrees);

  senderExp.join();
  recverExp.join();

  Matrix ldpc = generate_ldpc(numLeaves, numTrees);
  std::thread recverMult(mult_recver_gpu, ldpc, d_multiPprf, nonZeroRows, numTrees);

  recverMult.join();

  // assert(memcmp(&tree[numLeaves - 1], &tree2[numLeaves - 1], sizeof(*tree) * numLeaves) == 0);
  // assert(memcmp(&tree2[numLeaves - 1], senderLeaves, sizeof(*tree) * numLeaves) == 0);

  free(root);
  // free(tree2);

  return EXIT_SUCCESS;
}
