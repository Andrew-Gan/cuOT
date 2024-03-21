#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_matrix.h"
#include "base_ot.h"
#include "expand.h"
#include "lpn.h"

// number of gpu used per party
#define NGPU 2

class SilentOTSender;
class SilentOTRecver;

struct RCOTConfig {
  int id, logOT;
  uint64_t nTree;
  BaseOTType baseOT;
  ExpandType expander;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType compressor;
  uint64_t *choices;
  int ngpuAvail;
};

class SilentOT {
public:
  Role mRole;
  RCOTConfig mConfig;
  uint64_t depth, numOT, numLeaves;
  Expand *expander;
  Lpn *lpn;
  // base OT
  std::vector<Mat> m0;
  std::vector<Mat> m1;
  // pprf expansion
  Mat separated;
  Mat *buffer;
  // lpn compression
  Mat b64;

  SilentOT(RCOTConfig config) : mConfig(config) {
    depth = mConfig.logOT - std::log2(mConfig.nTree) + 0;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, depth);
  }
  virtual void base_ot() = 0;
  virtual void pprf_expand() = 0;
  virtual void lpn_compress() = 0;
};

class SilentOTSender : public SilentOT {
public:
  Mat *fullVector;
  blk *delta;
  std::vector<cudaEvent_t> expandEvents;

  SilentOTSender(RCOTConfig config);
  virtual ~SilentOTSender();
  virtual void base_ot();
  virtual void pprf_expand();
  virtual void lpn_compress();
};

class SilentOTRecver : public SilentOT {
public:
  Mat *puncVector;
  Mat choiceVector;
  uint64_t *puncPos;
  SilentOTSender *other = nullptr;
  std::vector<Mat> mc;
  std::atomic<bool> expandReady = false;
  uint64_t *activeParent;

  SilentOTRecver(RCOTConfig config);
  virtual ~SilentOTRecver();
  virtual void base_ot();
  virtual void get_punctured_key();
  virtual void pprf_expand();
  virtual void lpn_compress();
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

#endif
