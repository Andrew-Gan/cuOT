#include <assert.h>
#include <future>
#include "pprf.h"
#include "hash.h"
#include "rand.h"
#include "protocols.h"
#include "unit_test.h"

class SilentOT {
  
}

void silentOT(TreeNode root, uint64_t *choices, int depth, int numTrees) {
  int numOT = pow(2, depth + 7 - 1);

  auto senderTreeExp = std::async(pprf_sender, root, depth, numTrees);
  auto recverTreeExp = std::async(pprf_recver, choices, depth, numTrees);
  auto [fullVec_d, delta] = senderTreeExp.get();
  auto [puncVec_d, choiceVec_d] = recverTreeExp.get();

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

  // test_cot(fullVec_d, puncVec_d, choiceVec_d, delta);
  del_rand();
}
