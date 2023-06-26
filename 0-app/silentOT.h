#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <curand_kernel.h>
#include <vector>
#include <array>
#include <atomic>
#include "gpu_block.h"

#define CHUNK_SIDE (1<<18)

class SilentOT {
public:
  enum Role { Sender, Recver };
  SilentOT(int myid, int logOT, int numTrees);
  virtual std::pair<GPUBlock, GPUBlock> run() = 0;

  // network
  std::atomic<bool> msgDelivered = false;
  std::vector<std::vector<GPUBlock>> leftHash;
  std::vector<std::vector<GPUBlock>> rightHash;

protected:
  uint64_t id, depth, nTree, numOT;

  curandGenerator_t prng;
  Matrix randMatrix;
  SilentOT *other = nullptr;
  virtual void baseOT() = 0;
  virtual void expand() = 0;
};

class SilentOTSender : public SilentOT {
public:
  std::pair<GPUBlock, GPUBlock> run();
  SilentOTSender(int myid, int logOT, int numTrees);

protected:
  GPUBlock fullVector, delta;
  void baseOT();
  void expand();
  void compress(GPUBlock &fullVectorHashed, Matrix &randMatrix, GPUBlock &fullVector, int chunkC);
};

class SilentOTRecver : public SilentOT {
public:
  std::pair<GPUBlock, GPUBlock> run();
  SilentOTRecver(int myid, int logOT, int numTrees, uint64_t *mychoices);

protected:
  GPUBlock puncVector;
  uint64_t *choices;
  std::vector<std::vector<GPUBlock>> choiceHash;
  void baseOT();
  void expand();
  void compress(GPUBlock &puncVectorHashed, GPUBlock &choiceVectorHashed,
  Matrix &randMatrix, GPUBlock &puncVector, SparseVector &choiceVector,
  int chunkR, int chunkC);
};

extern std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

#endif
