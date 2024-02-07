#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "base_ot.h"
#include "expand.h"
#include "lpn.h"

#define NGPU 1

class SilentOTSender;
class SilentOTRecver;

struct SilentOTConfig {
  int id, logOT, nTree;
  BaseOTType baseOT;
  ExpandType expander;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType compressor;
  uint64_t *choices;
};

class SilentOT {
public:
  SilentOTConfig mConfig;
  uint64_t depth, numOT, numLeaves;
  Expand *expander[NGPU];
  Lpn *lpn[NGPU];
  std::vector<Vec> m0[NGPU];
  std::vector<Vec> m1[NGPU];
  
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
  Vec fullVector[NGPU];
  blk *delta[NGPU];
  SilentOTRecver *other = nullptr;
  std::vector<cudaEvent_t> expandEvents[NGPU];

  SilentOTSender(SilentOTConfig config);
  virtual ~SilentOTSender();
  virtual void base_ot();
  virtual void pprf_expand();
  virtual void lpn_compress();
};

class SilentOTRecver : public SilentOT {
public:
  Vec puncVector[NGPU];
  uint64_t *choiceVector;
  SilentOTSender *other = nullptr;
  std::vector<Vec> mc[NGPU];
  std::atomic<bool> expandReady = false;
  uint64_t *activeParent[2];

  SilentOTRecver(SilentOTConfig config);
  virtual ~SilentOTRecver();
  virtual void base_ot();
  virtual void pprf_expand();
  virtual void lpn_compress();
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

#endif
