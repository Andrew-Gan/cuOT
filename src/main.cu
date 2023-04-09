#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <future>
#include <random>

#include "pprf_cpu.h"
#include "pprf_gpu.h"
#include "ldpc.h"
#include "gemm_cpu.h"
#include "gemm_gpu.h"

uint64_t* genChoices(int numTrees, int numLeaves) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void testCpu(TreeNode root, uint64_t *choices, int depth, int numTrees, int numLeaves) {
  auto senderExp = std::async(pprf_sender_cpu, choices, root, depth, numTrees);
  auto recverExp = std::async(pprf_recver_cpu, choices, depth, numTrees);
  auto [fullVec, delta] = senderExp.get();
  auto [puncVec, puncIndices] = recverExp.get();

  // printf("Punctured at: ");
  // for(int i = 0; i < numLeaves; i++) {
  //   if (memcmp(&fullVec[i], &puncVec[i], sizeof(*puncVec)) != 0)
  //     printf("%d ", i);
  // }
  // printf("\n");

  Matrix ldpc = generate_ldpc(numLeaves, numTrees);
  std::thread recverMult(mult_recver_cpu, ldpc, puncIndices, numTrees);
  recverMult.join();
}

void testGpu(TreeNode root, uint64_t *choices, int depth, int numTrees, int numLeaves) {
  auto senderExp = std::async(pprf_sender_gpu, choices, root, depth, numTrees);
  auto recverExp = std::async(pprf_recver_gpu, choices, depth, numTrees);
  auto [d_fullVec, delta] = senderExp.get();
  auto [d_puncVec, puncIndices] = recverExp.get();

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

  Matrix ldpc = generate_ldpc(numLeaves, numTrees);
  std::thread recverMult(mult_recver_gpu, ldpc, puncIndices, numTrees);
  recverMult.join();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./pprf d t\n");
    return EXIT_FAILURE;
  }

  size_t depth = atoi(argv[1]);
  int numTrees = atoi(argv[2]);
  size_t numLeaves = pow(2, depth);
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  printf("Depth: %lu, OTs: %lu, Weight: %d\n", depth, numLeaves, numTrees);

  uint64_t *choices = genChoices(numTrees, numLeaves);
  testCpu(root, choices, depth, numTrees, numLeaves);
  testGpu(root, choices, depth, numTrees, numLeaves);

  delete choices;

  return EXIT_SUCCESS;
}
