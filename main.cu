// seed gen -> seed expansion -> matrix gen -> random matrix hashing -> cot

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <future>
#include <random>

#include "pprf.h"
#include "rand.h"
#include "hash.h"
#include "unit_test.h"

void cuda_check() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    fprintf(stderr, "There is no device.\n");
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major >= 1)
      break;
  }
  if (dev == deviceCount)
    fprintf(stderr, "There is no device supporting CUDA.\n");
  else
    cudaSetDevice(dev);
}

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = (uint64_t*) malloc(sizeof(uint64_t) * numTrees);
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void run(TreeNode root, uint64_t *choices, int depth, int numTrees, size_t numOT) {
  struct timespec expStart, hashStart, end;
  float expDuration = 0, hashDuration = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    clock_gettime(CLOCK_MONOTONIC, &expStart);

    auto senderExp = std::async(pprf_sender, choices, root, depth, numTrees);
    auto recverExp = std::async(pprf_recver, choices, depth, numTrees);
    auto [d_fullVec, delta] = senderExp.get();
    auto [d_puncVec, d_choiceVec] = recverExp.get();

    clock_gettime(CLOCK_MONOTONIC, &hashStart);

    if (numOT < CHUNK_SIDE) {
      Matrix d_randMatrix = gen_rand(2 * numOT, numOT); // transposed
      std::thread senderHash(hash_sender, d_randMatrix, d_fullVec, 0);
      std::thread recverHash(hash_recver, d_randMatrix, d_choiceVec, d_puncVec, 0);
      senderHash.join();
      recverHash.join();
    }
    else {
      for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
        for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
          Matrix d_randMatrix = gen_rand(CHUNK_SIDE, CHUNK_SIDE);
          std::thread senderHash(hash_sender, d_randMatrix, d_fullVec, chunkC);
          std::thread recverHash(hash_recver, d_randMatrix, d_choiceVec, d_puncVec, chunkC);
          senderHash.join();
          recverHash.join();
        }
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

#ifdef DEBUG_MODE
    test_cot(d_fullVec, d_puncVec, d_choiceVec, delta);
#endif

    expDuration += (hashStart.tv_sec - expStart.tv_sec) * 1000;
    expDuration += (hashStart.tv_nsec - expStart.tv_nsec) / 1000000.0;
    hashDuration += (end.tv_sec - hashStart.tv_sec) * 1000;
    hashDuration += (end.tv_nsec - hashStart.tv_nsec) / 1000000.0;
  }

  del_rand();
  printf("Seed exp using GPU: %0.4f ms\n", expDuration / NUM_SAMPLES);
  printf("chunk = %d x %d\n", 2 * numOT / CHUNK_SIDE, numOT / CHUNK_SIDE);
  printf("Matrix hash using GPU: %0.4f ms\n\n", hashDuration / NUM_SAMPLES);
}

int main(int argc, char** argv) {
  cuda_check();
#ifdef DEBUG_MODE
  // test_rsa();
  test_aes();
#endif

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

  uint64_t *choices = gen_choices(numTrees);
  run(root, choices, actualDepth, numTrees, numOT);

  free(choices);

  return EXIT_SUCCESS;
}
