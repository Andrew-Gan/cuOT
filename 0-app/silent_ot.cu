#include <future>
#include "pprf.h"
#include "hash.h"
#include "rand.h"
#include "silent_ot.h"

SilentOT::SilentOT(Role role, int id, int logOT, int numTrees) : OT(role, id) {
  nTree = numTrees;
  depth = pow(2, logOT + 7 - 1);
  numOT = pow(2, logOT);
  if (role == Sender) {
    while(recvers[id] == nullptr);
    OT *recv = recvers[id];
    other = dynamic_cast<SilentOT*>(recv);
  }
  else {
    while(senders[id] == nullptr);
    OT *send = senders[id];
    other = dynamic_cast<SilentOT*>(send);
  }
}

void SilentOT::send(GPUBlock &m0, GPUBlock &m1) {
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;
  std::future senderTreeExp = std::async(pprf_sender, root, depth, nTree);
  auto [fullVec_d, delta] = senderTreeExp.get();
  if (numOT < CHUNK_SIDE) {
    Matrix randMatrix_d = gen_rand(2 * numOT, numOT); // transposed
    std::future senderMatrixHash = std::async(hash_sender, randMatrix_d, fullVec_d, 0);
    senderMatrixHash.get();
  }
  else {
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        Matrix randMatrix_d = gen_rand(CHUNK_SIDE, CHUNK_SIDE);
        std::future senderMatrixHash = std::async(hash_sender, randMatrix_d, fullVec_d, chunkC);
        senderMatrixHash.get();
      }
    }
  }
  del_rand();
}

GPUBlock SilentOT::recv(uint8_t choice) {
  return recv((uint64_t*) &choice);
}

GPUBlock SilentOT::recv(uint64_t *choices) {
  std::future recverTreeExp = std::async(pprf_recver, choices, depth, nTree);
  auto [puncVec_d, choiceVec_d] = recverTreeExp.get();
  if (numOT < CHUNK_SIDE) {
    Matrix randMatrix_d = gen_rand(2 * numOT, numOT); // transposed
    std::future recverMatrixHash = std::async(hash_recver, randMatrix_d, choiceVec_d, puncVec_d, 0);
    recverMatrixHash.get();
  }
  else {
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        Matrix randMatrix_d = gen_rand(CHUNK_SIDE, CHUNK_SIDE);
        std::future recverMatrixHash = std::async(hash_recver, randMatrix_d, choiceVec_d, puncVec_d, chunkC);
        recverMatrixHash.get();
      }
    }
  }
  del_rand();
  return GPUBlock();
}
