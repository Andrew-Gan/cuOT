#include "pprf.h"
#include "hash.h"
#include "rand.h"
#include "simplest_ot.h"
#include "silent_ot.h"
#include <future>

std::array<std::atomic<SilentOT*>, 100> silentOTSenders;
std::array<std::atomic<SilentOT*>, 100> silentOTRecvers;

SilentOT::SilentOT(Role myrole, int myid, int logOT, int numTrees, uint64_t *mychoices) {
  role = myrole;
  id = myid;
  choices = mychoices;

  if (role == Sender) {
    silentOTSenders[id] = this;
    while(silentOTRecvers[id] == nullptr);
    other = silentOTRecvers[id];
  }
  else {
    silentOTRecvers[id] = this;
    while(silentOTSenders[id] == nullptr);
    other = silentOTSenders[id];
  }

  nTree = numTrees;
  depth = logOT - log2(numTrees) + 1;
  numOT = pow(2, logOT);
}

void SilentOT::sendBaseOTs() {
  std::vector<std::future<std::array<std::vector<GPUBlock>, 2>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Sender, t).send(depth);
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
}

void SilentOT::recvBaseOTs() {
  std::vector<std::future<std::vector<GPUBlock>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Recver, t).recv(depth, rand());
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

std::pair<GPUBlock, GPUBlock> SilentOT::send() {
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  auto [fullVector, delta] = pprf_sender(root, depth, nTree);
  GPUBlock fullVectorHashed(numOT * TREENODE_SIZE);
  return std::pair<GPUBlock, GPUBlock>(); //debug

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
  return std::make_pair(fullVectorHashed, delta);
}

std::pair<GPUBlock, GPUBlock> SilentOT::recv() {
  auto [puncVector, choiceVector] = pprf_recver(choices, depth, nTree);
  GPUBlock puncVectorHashed(numOT * TREENODE_SIZE);
  GPUBlock choiceVectorHashed(numOT * TREENODE_SIZE);
  return std::pair<GPUBlock, GPUBlock>(); //debug

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
  return std::make_pair(puncVectorHashed, choiceVectorHashed);
}
