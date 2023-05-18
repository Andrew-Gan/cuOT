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

void SilentOT::send(GPUBlock &m0, GPUBlock &m1) {
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;
  auto [fullVec_d, delta] = pprf_sender(root, depth, nTree);
  if (numOT < CHUNK_SIDE) {
    EventLog::start(MatrixInit);
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    EventLog::end(MatrixInit);
    hash_sender(randMatrix, fullVec_d, 0);
  }
  else {
    EventLog::start(MatrixInit);
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    EventLog::end(MatrixInit);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        EventLog::start(MatrixInit);
        gen_rand(prng, randMatrix);
        EventLog::end(MatrixInit);
        hash_sender(randMatrix, fullVec_d, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
}

GPUBlock SilentOT::recv(uint8_t choice) {
  return recv((uint64_t*) &choice);
}

GPUBlock SilentOT::recv(uint64_t *choices) {
  GPUBlock mb(1024);
  auto [puncVec_d, choiceVec_d] = pprf_recver(choices, depth, nTree);
  if (numOT < CHUNK_SIDE) {
    EventLog::start(MatrixInit);
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    EventLog::end(MatrixInit);
    hash_recver(randMatrix, choiceVec_d, puncVec_d, 0);
  }
  else {
    EventLog::start(MatrixInit);
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    EventLog::end(MatrixInit);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        EventLog::start(MatrixInit);
        gen_rand(prng, randMatrix);
        EventLog::end(MatrixInit);
        hash_recver(randMatrix, choiceVec_d, puncVec_d, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
  return mb;
}
