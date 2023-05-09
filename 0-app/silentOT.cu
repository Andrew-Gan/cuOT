#include <assert.h>
#include <future>
#include "pprf.h"
#include "hash.h"
#include "rand.h"
#include "protocols.h"
#include "unit_test.h"

void silentOT(TreeNode root, uint64_t *choices, int depth, int numTrees) {
  struct timespec expStart, hashStart, end;
  float expDuration = 0, hashDuration = 0;
  int numOT = pow(2, depth + 7 - 1);

  for (int i = 0; i < NUM_SAMPLES; i++) {
    clock_gettime(CLOCK_MONOTONIC, &expStart);

    auto senderTreeExp = std::async(pprf_sender, root, depth, numTrees);
    auto recverTreeExp = std::async(pprf_recver, choices, depth, numTrees);
    auto [fullVec_d, delta] = senderTreeExp.get();
    auto [puncVec_d, choiceVec_d] = recverTreeExp.get();

    clock_gettime(CLOCK_MONOTONIC, &hashStart);

    if (numOT < CHUNK_SIDE) {
      Matrix randMatrix_d = gen_rand(2 * numOT, numOT); // transposed
      std::thread senderMatrixHash(hash_sender, randMatrix_d, fullVec_d, 0);
      std::thread recverMatrixHash(hash_recver, randMatrix_d, choiceVec_d, puncVec_d, 0);
      senderMatrixHash.join();
      recverMatrixHash.join();
    }
    else {
      for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
        for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
          Matrix randMatrix_d = gen_rand(CHUNK_SIDE, CHUNK_SIDE);
          std::thread senderMatrixHash(hash_sender, randMatrix_d, fullVec_d, chunkC);
          std::thread recverMatrixHash(hash_recver, randMatrix_d, choiceVec_d, puncVec_d, chunkC);
          senderMatrixHash.join();
          recverMatrixHash.join();
        }
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    test_cot(fullVec_d, puncVec_d, choiceVec_d, delta);

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
