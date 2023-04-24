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
  uint64_t *choices = (uint64_t*) malloc(sizeof(uint64_t) * numTrees);
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

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  if (numOT < CHUNK_SIDE) {
    Matrix d_randMatrix = gen_rand_gpu(2 * numOT, numOT); // transposed
    std::thread senderMult(mult_sender_gpu, d_randMatrix, d_fullVec, 0);
    std::thread recverMult(mult_recver_gpu, d_randMatrix, d_choiceVec, d_puncVec, 0);
    senderMult.join();
    recverMult.join();
    cudaFree(d_randMatrix.data);
  }
  else {
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        Matrix d_randMatrix = gen_rand_gpu(CHUNK_SIDE, CHUNK_SIDE);
        std::thread senderMult(mult_sender_gpu, d_randMatrix, d_fullVec, chunkC);
        std::thread recverMult(mult_recver_gpu, d_randMatrix, d_choiceVec, d_puncVec, chunkC);
        senderMult.join();
        recverMult.join();
        cudaFree(d_randMatrix.data);
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Matrix mult using GPU: %0.4f ms\n", duration / NUM_SAMPLES);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: ./pprf d t\n");
    return EXIT_FAILURE;
  }

  int userDepth = atoi(argv[1]);
  size_t numOT = pow(2, userDepth);
  // each node has 2^7 bits
  // num bits in final layer = 2 * OT, to be halved during encoding
  size_t actualDepth = userDepth - 7 + 1;
  int numTrees = atoi(argv[2]);
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  printf("OTs: %lu, Trees: %d\n", numOT, numTrees);

  uint64_t *choices = genChoices(numTrees);
  testCpu(root, choices, actualDepth, numTrees, numOT);
  testGpu(root, choices, actualDepth, numTrees, numOT);

  free(choices);

  return EXIT_SUCCESS;
}
