#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "base_ot.h"
#include "expand.h"
#include "compress.h"

#define CHUNK_SIDE (1<<18)

class SilentOTSender;
class SilentOTRecver;

struct SilentOTConfig {
  int id, logOT, nTree;
  BaseOTType baseOT;
  ExpandType expander;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  CompressType compressor;
  // recver only
  uint64_t *choices;
};

class SilentOT {
public:
  SilentOTConfig mConfig;
  uint64_t depth, numOT, numLeaves;
  
  SilentOT(SilentOTConfig config) : mConfig(config) {
    depth = mConfig.logOT - log2((float) mConfig.nTree) + 1;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, depth);
  }
  virtual void base_ot() = 0;
  virtual void pprf_expand() = 0;
  virtual void lpn_compress() = 0;
};

class SilentOTSender : public SilentOT {
public:
  Vec fullVector;
  blk *delta = nullptr;
  SilentOTRecver *other = nullptr;
  std::vector<Vec> leftHash;
  std::vector<Vec> rightHash;
  std::vector<cudaEvent_t> expandEvents;

  SilentOTSender(SilentOTConfig config);
  virtual void base_ot();
  virtual void pprf_expand();
  virtual void lpn_compress();
};

class SilentOTRecver : public SilentOT {
public:
  Vec puncVector, choiceVector;
  std::vector<Vec> leftBuffer;
  std::vector<Vec> rightBuffer;
  std::atomic<bool> eventsRecorded = false;
  SilentOTSender *other = nullptr;
  std::vector<Vec> choiceHash;

  SilentOTRecver(SilentOTConfig config);
  virtual void base_ot();
  virtual void pprf_expand();
  virtual void lpn_compress();
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

#endif
