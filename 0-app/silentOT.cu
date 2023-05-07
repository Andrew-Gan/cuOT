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

    auto senderExp = std::async(pprf_sender, root, depth, numTrees);
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

#ifdef UNIT_TEST
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
