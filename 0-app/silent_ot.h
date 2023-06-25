#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <curand_kernel.h>
#include <vector>
#include <atomic>
#include "gpu_block.h"

#define CHUNK_SIDE (1<<18)

class SilentOT {
public:
  enum Role { Sender, Recver };
  SilentOT(Role myrole, int myid, int logOT, int numTrees, uint64_t *mychoices);
  std::pair<GPUBlock, GPUBlock> send();
  std::pair<GPUBlock, GPUBlock> recv();

private:
  Role role;
  uint64_t id, depth, nTree, numOT;
  curandGenerator_t prng;
  Matrix randMatrix;
  SilentOT *other = nullptr;

  // network
  std::atomic<uint64_t> msgDelivered = 0;
  std::vector<std::vector<GPUBlock>> leftHash;
  std::vector<std::vector<GPUBlock>> rightHash;

  // sender only
  GPUBlock fullVector, delta;
  void baseOT_send();
  void pprf_send();
  void hash_sender(GPUBlock &fullVectorHashed, Matrix &randMatrix, GPUBlock &fullVector, int chunkC);

  // recver only
  GPUBlock puncVector;
  uint64_t *choices;
  std::vector<std::vector<GPUBlock>> choiceHash;
  void baseOT_recv();
  void pprf_recv();
  void hash_recver(GPUBlock &puncVectorHashed, GPUBlock &choiceVectorHashed,
  Matrix &randMatrix, GPUBlock &puncVector, SparseVector &choiceVector,
  int chunkR, int chunkC);
};

extern std::array<std::atomic<SilentOT*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOT*>, 100> silentOTRecvers;

#endif
