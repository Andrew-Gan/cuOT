#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <future>
#include <random>

#include "pprf_cpu.h"
#include "pprf_gpu.h"
#include "rand_cpu.h"
#include "rand_gpu.h"
#include "gemm_cpu.h"
#include "gemm_gpu.h"

uint64_t* genChoices(int numTrees) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void testCpu(TreeNode root, uint64_t *choices, int depth, int numTrees, size_t numOT) {
  // int numLeaves = numOT / (8 * TREENODE_SIZE);
  // auto senderExp = std::async(pprf_sender_cpu, choices, root, depth, numTrees);
  // auto recverExp = std::async(pprf_recver_cpu, choices, depth, numTrees);
  // auto [fullVec, delta] = senderExp.get();
  // auto [puncVec, d_choiceVec] = recverExp.get();

  // // printf("Punctured at: ");
  // // for(int i = 0; i < numLeaves; i++) {
  // //   if (memcmp(&fullVec[i], &puncVec[i], sizeof(*puncVec)) != 0)
  // //     printf("%d ", i);
  // // }
  // // printf("\n");

  // Matrix ldpc = generate_(numLeaves, numTrees);
  // printf("ldpc: %d x %d\n", ldpc.rows, ldpc.cols);
  // std::thread recverMult(mult_recver_cpu, ldpc, d_choiceVec, numTrees);
  // recverMult.join();
}

void testGpu(TreeNode root, uint64_t *choices, int depth, int numTrees, size_t numOT) {
  auto senderExp = std::async(pprf_sender_gpu, choices, root, depth, numTrees);
  auto recverExp = std::async(pprf_recver_gpu, choices, depth, numTrees);
  auto [d_fullVec, delta] = senderExp.get();
  auto [d_puncVec, d_choiceVec] = recverExp.get();

  // int numLeaves = 2 * numOT / (8 * TREENODE_SIZE);
  // TreeNode *puncVec = (TreeNode*) malloc(numLeaves * sizeof(*puncVec));
  // cudaMemcpy(puncVec, d_puncVec, numLeaves * sizeof(*d_puncVec), cudaMemcpyDeviceToHost);
  // TreeNode *fullVec = (TreeNode*) malloc(numLeaves * sizeof(*fullVec));
  // cudaMemcpy(fullVec, d_fullVec, numLeaves * sizeof(*d_fullVec), cudaMemcpyDeviceToHost);
  // printf("Punctured at: ");
  // for(int i = 0; i < numLeaves; i++) {
  //   if (memcmp(&fullVec[i], &puncVec[i], sizeof(*puncVec)) != 0)
  //     printf("%d ", i);
  // }
  // printf("\n");

  Matrix d_randMatrix = gen_rand_gpu(numOT, 2 * numOT);
  std::thread senderMult(mult_sender_gpu, d_randMatrix, d_fullVec);
  std::thread recverMult(mult_recver_gpu, d_randMatrix, d_choiceVec, d_puncVec);
  senderMult.join();
  recverMult.join();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./pprf d t\n");
    return EXIT_FAILURE;
  }

  // each node has 2^4 bytes or 2^7 bits
  // num bits in final layer = 2 * OT, will be halved during encoding
  size_t depth = atoi(argv[1]) - 7 + 1;
  int numTrees = atoi(argv[2]);
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  size_t numOT = pow(2, depth+7);
  printf("Depth: %lu, OTs: %lu, Weight: %d\n", depth+7, numOT, numTrees);

  uint64_t *choices = genChoices(numTrees);
  testCpu(root, choices, depth, numTrees, numOT);
  testGpu(root, choices, depth, numTrees, numOT);

  delete choices;

  return EXIT_SUCCESS;
}
