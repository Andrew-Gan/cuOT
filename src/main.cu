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
#include "gemm_cpu.h"
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
    fprintf(stderr, "Usage: ./pprf log(n) t\n");
    return EXIT_FAILURE;
  }

  size_t depth = atoi(argv[1]);
  int numTrees = atoi(argv[2]);
  size_t numNodes = pow(2, depth + 1) - 1;
  size_t numLeaves = numNodes / 2 + 1;
  TreeNode *root = (TreeNode*) malloc(sizeof(*root));
  root->data[0] = 123456;
  root->data[1] = 7890123;

  printf("Depth: %lu, Nodes: %lu, Threads: %d\n", depth, numNodes, numTrees);

  TreeNode *sparseVec = (TreeNode*) malloc(numLeaves * sizeof(*sparseVec));
  int *nonZerosCpu = (int*) malloc(numTrees * sizeof(*nonZerosCpu));
  std::thread senderExpCpu(pprf_sender_cpu, root, depth, aesni_init_ctx, aesni_ecb_encrypt, numTrees);
  std::thread recverExpCpu(pprf_recver_cpu, aesni_init_ctx, aesni_ecb_encrypt, sparseVec, nonZerosCpu, depth, numTrees);
  senderExpCpu.join();
  recverExpCpu.join();

  TreeNode *d_sparseVec;
  cudaMalloc(&d_sparseVec, sizeof(*d_sparseVec) * numLeaves);
  int *nonZerosGpu = (int*) malloc(numTrees * sizeof(*nonZerosGpu));
  std::thread senderExpGpu(pprf_sender_gpu, root, depth, numTrees);
  std::thread recverExpGpu(pprf_recver_gpu, d_sparseVec, nonZerosGpu, depth, numTrees);
  senderExpGpu.join();
  recverExpGpu.join();

  TreeNode *gpu_sparseVec = (TreeNode*) malloc(numLeaves * sizeof(*gpu_sparseVec));
  cudaMemcpy(gpu_sparseVec, d_sparseVec, numLeaves * sizeof(*d_sparseVec), cudaMemcpyDeviceToHost);

  // assert(memcpy(gpu_sparseVec, sparseVec, numLeaves * sizeof(*d_sparseVec)) == 0);

  TreeNode zeroNode;
  memset(&zeroNode, 0, sizeof(zeroNode));
  // printf("cpu non-zero at: ");
  // for(int i = 0; i < numLeaves; i++) {
  //   if (memcmp(&sparseVec[i], &zeroNode, sizeof(zeroNode)) != 0) {
  //     printf("%d ", i);
  //   }
  // }
  // printf("\n");
  // printf("gpu non-zero at: ");
  // for(int i = 0; i < numLeaves; i++) {
  //   if (memcmp(&gpu_sparseVec[i], &zeroNode, sizeof(zeroNode)) != 0) {
  //     printf("%d ", i);
  //   }
  // }
  // printf("\n");

  Matrix ldpc = generate_ldpc(numLeaves, numTrees);


  std::thread recverMultCpu(mult_recver_cpu, ldpc, sparseVec, nonZerosGpu, numTrees);
  recverMultCpu.join();
  std::thread recverMultGpu(mult_recver_gpu, ldpc, d_sparseVec, nonZerosGpu, numTrees);
  recverMultGpu.join();

  free(root);
  // free(tree2);

  return EXIT_SUCCESS;
}
