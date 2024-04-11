#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_matrix.h"
#include "base_ot.h"
#include "pprf.h"
#include "quasi_cyclic.h"

// number of gpu used per party
#define NGPU 1

class SilentOTSender;
class SilentOTRecver;

struct SilentConfig {
  int id, logOT;
  uint64_t nTree;
  BaseOTType baseOT;
  PprfType pprf;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType dualLPN;
  uint64_t *choices;
};

class SilentOT {
public:
  Role mRole;
  int mDev;
  SilentConfig mConfig;
  uint64_t depth, numOT, numLeaves;
  Pprf *expander;
  QuasiCyclic *lpn;
  // base OT
  std::vector<Mat> m0;
  std::vector<Mat> m1;

  SilentOT(SilentConfig config) : mConfig(config) {
    depth = mConfig.logOT - std::log2(mConfig.nTree) + 0;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, depth);
  }
  virtual void base_ot() = 0;
  virtual void seed_expand() = 0;
  virtual void dual_lpn() = 0;

protected:
  // pprf expansion
  Mat separated;
  Mat *buffer;
  // lpn compression
  Mat b64;
};

class SilentOTSender : public SilentOT {
public:
  Mat *fullVector;
  blk *delta;

  SilentOTSender(SilentConfig config);
  virtual ~SilentOTSender();
  virtual void base_ot();
  virtual void seed_expand();
  virtual void dual_lpn();
};

class SilentOTRecver : public SilentOT {
public:
  Mat *puncVector;
  Mat choiceVector;
  uint64_t *puncPos;
  SilentOTSender *other = nullptr;
  std::vector<Mat> mc;
  uint64_t *activeParent;

  SilentOTRecver(SilentConfig config);
  virtual ~SilentOTRecver();
  virtual void base_ot();
  virtual void get_punctured_key();
  virtual void seed_expand();
  virtual void dual_lpn();

private:
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 16> silentOTRecvers;

#endif
