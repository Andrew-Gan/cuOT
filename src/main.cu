#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "aes.h"
#include "aesni.h"
#include "pprf_cpu.h"
#include "pprf_gpu.h"
#include <math.h>
#include <thread>

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
  TreeNode *tree = (TreeNode*) malloc(sizeof(*tree) * numNodes);
  tree[0].data[0] = 123456;
  tree[0].data[1] = 7890123;

  TreeNode *tree2 = (TreeNode*) malloc(sizeof(*tree) * numNodes);
  memcpy(tree2, tree, sizeof(*tree) * numNodes);

  TreeNode *senderLeaves, *recverLeaves;
  cudaMallocHost(&senderLeaves, sizeof(*senderLeaves) * numLeaves);
  cudaMallocHost(&recverLeaves, sizeof(*recverLeaves) * numLeaves);

  printf("Depth: %lu, Nodes: %lu, Threads: %d\n", depth, numNodes, numTrees);
  aescpu_tree_expand(tree, depth, aes_init_ctx, aes_ecb_encrypt, "AES", numTrees);
  aescpu_tree_expand(tree2, depth, aesni_init_ctx, aesni_ecb_encrypt, "AESNI", numTrees);

  std::thread sender(pprf_sender_gpu, tree, senderLeaves, depth, numTrees);
  std::thread recver(pprf_recver_gpu, recverLeaves, depth, numTrees);

  sender.join();
  recver.join();

  assert(memcmp(&tree[numLeaves - 1], &tree2[numLeaves - 1], sizeof(*tree) * numLeaves) == 0);
  // assert(memcmp(&tree2[numLeaves - 1], senderLeaves, sizeof(*tree) * numLeaves) == 0);

  printf("Punctured at nodes: ");
  for (int i = 0; i < numLeaves; i++) {
    if (memcmp(&senderLeaves[i], &recverLeaves[i], sizeof(TreeNode)) != 0) {
      printf("%d ", i);
    }
  }
  printf("\n");

  // print_leaves(senderLeaves, numLeaves);
  // print_leaves(recverLeaves, numLeaves);

  free(tree);
  free(tree2);
  cudaFreeHost(senderLeaves);
  cudaFreeHost(recverLeaves);

  return EXIT_SUCCESS;
}
