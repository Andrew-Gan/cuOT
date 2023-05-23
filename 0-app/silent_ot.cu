#include "pprf.h"
#include "hash.h"
#include "rand.h"
#include "silent_ot.h"

SilentOT::SilentOT(Role myrole, int myid, int logOT, int numTrees) : OT(myrole, myid) {
  nTree = numTrees;
  depth = logOT - 7 + 1;
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

std::pair<GPUBlock, GPUBlock> SilentOT::send() {
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;
  auto [fullVector, delta] = pprf_sender(root, depth, nTree);
  GPUBlock fullVectorHashed(numOT / 8);

  if (numOT < CHUNK_SIDE) {
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    hash_sender(fullVectorHashed, randMatrix, fullVector, 0);
  }
  else {
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        gen_rand(prng, randMatrix);
        hash_sender(fullVectorHashed, randMatrix, fullVector, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
  return std::make_pair(fullVector, delta);
}

std::pair<GPUBlock, SparseVector> SilentOT::recv(uint64_t *choices) {
  auto [puncVector, choiceVector] = pprf_recver(choices, depth, nTree);
  GPUBlock puncVectorHashed(numOT / 8);
  GPUBlock choiceVectorHashed(numOT / 8);

  if (numOT < CHUNK_SIDE) {
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    hash_recver(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, 0, 0);
  }
  else {
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        gen_rand(prng, randMatrix);
        hash_recver(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, chunkR, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
  return std::make_pair(puncVector, choiceVector);
}
