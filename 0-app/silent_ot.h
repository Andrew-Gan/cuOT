#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <curand_kernel.h>
#include <vector>
#include <array>
#include <atomic>
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "quasi_cyclic.h"
#include "aes.h"

#define CHUNK_SIDE (1<<18)

class SilentOTSender;
class SilentOTRecver;

class SilentOT {
public:
  enum Role { Sender, Recver };
  SilentOT(int myid, int logOT, int numTrees) : id(myid), nTree(numTrees) {
    depth = logOT - log2((float) nTree) + 1;
    numOT = pow(2, logOT);
    numLeaves = pow(2, depth);
  }
  virtual void run() = 0;

  // network
  std::vector<GPUvector<OTblock>> leftHash;
  std::vector<GPUvector<OTblock>> rightHash;

protected:
  Aes aesLeft, aesRight;
  GPUvector<OTblock> bufferA, bufferB;
  GPUvector<OTblock> leftNodes, rightNodes;
  uint64_t id, depth, nTree, numOT, numLeaves;
  virtual void baseOT() = 0;
  virtual void expand() = 0;
};

class SilentOTSender : public SilentOT {
public:
  SilentOTSender(int myid, int logOT, int numTrees);
  void run();

private:
  GPUvector<OTblock> fullVector;
  OTblock *delta = nullptr;
  SilentOTRecver *other = nullptr;
  void baseOT();
  void buffer_init();
  void expand();
};

class SilentOTRecver : public SilentOT {
public:
  std::vector<cudaEvent_t> expandEvents;
  std::atomic<bool> eventsRecorded = false;
  SilentOTRecver(int myid, int logOT, int numTrees, uint64_t *mychoices);
  void run();

private:
  GPUvector<OTblock> puncVector, choiceVector;
  uint64_t *choices;
  SilentOTSender *other = nullptr;
  std::vector<GPUvector<OTblock>> choiceHash;
  void baseOT();
  void buffer_init();
  void expand();
  void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

#endif
