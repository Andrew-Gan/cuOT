#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_matrix.h"
#include "base_ot.h"
#include "pprf.h"
#include "quasi_cyclic.h"

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
  int gpuPerParty;
};

class SilentOT {
public:
  Role mRole;
  SilentConfig mConfig;
  uint64_t mDepth, numOT, numLeaves;
  Pprf *expander;
  QuasiCyclic *lpn;
  // base OT
  Mat m0, m1;

  SilentOT(SilentConfig config) : mConfig(config) {
    mDepth = mConfig.logOT - std::log2(mConfig.nTree) + 0;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, mDepth);
  }
  virtual ~SilentOT() {}
  virtual void base_ot() = 0;
  virtual void seed_expand() = 0;
  virtual void dual_lpn() = 0;

protected:
  // pprf expansion
  Mat separated;
  Mat *buffer;
};

class SilentOTSender : public SilentOT {
public:
  Mat *fullVector;
  blk *delta;
  static blk *m0_h, *m1_h;

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
  static blk *mc_h;
  Mat mc;
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
