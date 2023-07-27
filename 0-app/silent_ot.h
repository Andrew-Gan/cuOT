#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "base_ot.h"
#include "expander.h"
#include "compressor.h"

#define CHUNK_SIDE (1<<18)

class SilentOTSender;
class SilentOTRecver;

struct SilentOTConfig {
  int id, logOT, nTree;
  uint64_t *choices;
  BaseOTType baseOT;
  ExpanderType expander;
  CompressType compressor;
};

class SilentOT {
public:
  SilentOT(SilentOTConfig config) : mConfig(config) {
    depth = mConfig.logOT - log2((float) mConfig.nTree) + 1;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, depth);
  }
  virtual void run() = 0;

  // network
  std::vector<GPUvector<OTblock>> leftHash;
  std::vector<GPUvector<OTblock>> rightHash;

protected:
  SilentOTConfig mConfig;
  GPUvector<OTblock> bufferA, bufferB;
  GPUvector<OTblock> leftNodes, rightNodes;
  uint64_t depth, numOT, numLeaves;
  virtual void base_ot() = 0;
  virtual void pprf_expand() = 0;
};

class SilentOTSender : public SilentOT {
public:
  SilentOTSender(SilentOTConfig config);
  void run();
  std::pair<GPUvector<OTblock>, OTblock*> get() { return {fullVector, delta}; }

private:
  GPUvector<OTblock> fullVector;
  OTblock *delta = nullptr;
  SilentOTRecver *other = nullptr;
  GPUvector<OTblock> leftSum, rightSum;
  void base_ot();
  void buffer_init();
  void pprf_expand();
};

class SilentOTRecver : public SilentOT {
public:
  std::vector<cudaEvent_t> expandEvents;
  std::atomic<bool> eventsRecorded = false;
  SilentOTRecver(SilentOTConfig config);
  void run();
  std::array<GPUvector<OTblock>, 2> get() { return {puncVector, choiceVector}; }

private:
  GPUvector<OTblock> puncVector, choiceVector;
  SilentOTSender *other = nullptr;
  std::vector<GPUvector<OTblock>> choiceHash;
  void base_ot();
  void buffer_init();
  void pprf_expand();
  void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

#endif
